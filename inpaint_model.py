""" common model for DCGAN """
import logging
import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty, gan_hinge_loss
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gen_conv, gen_deconv, dis_conv, sn_dis_conv
from inpaint_ops import random_bbox, bbox2mask, local_patch
from inpaint_ops import spatial_discounting_mask
from inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net', exclusionmask=None):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        multires = config.MULTIRES
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)
        hasmask = exclusionmask is not None
        if hasmask:
            exclusionmask = tf.cast(tf.less(exclusionmask[:,:,:,0:1], 0.5), tf.float32)
            x = tf.concat([x, exclusionmask], axis=3)
        use_gating = config.GATING

        # two stage network
        cnum = 24 if use_gating else 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6', gating=use_gating)
            mask_s = resize_mask_like(mask, x)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12', gating=use_gating)
            x = gen_deconv(x, 2*cnum, name='conv13_upsample', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14', gating=use_gating)
            x = gen_deconv(x, cnum, name='conv15_upsample', gating=use_gating)
            x = gen_conv(x, cnum//2, 3, 1, name='conv16', gating=use_gating)
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            if hasmask:
                xnow = tf.concat([xnow, exclusionmask], axis=3)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1', gating=use_gating)
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous', gating=use_gating)
            x_hallu = x
            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1', gating=use_gating)
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                         activation=tf.nn.relu, gating=use_gating)
            flows = []
            use_attentionmask = hasmask and config.ATTENTION_MASK
            if use_attentionmask:
                ex_mask_s = resize_mask_like(exclusionmask, x)
            if multires: #scale down feature map, run contextual attention, scale up and paste inpainted region into original feature map
                logger.info('USING MULTIRES')
                logger.info('original x shape: ' + str(x.shape))
                logger.info('original mask shape: ' + str(mask_s.shape))
                x_multi = [x]
                mask_multi = [mask_s]
                if use_attentionmask:
                    exclusion_mask_multi = [ex_mask_s]
                for i in range(config.LEVELS-1):
                    #x = gen_conv(x, 4*cnum, 3, 2, name='pyramid_downsample_'+str(i+1))
                    x = resize(x, scale=0.5)
                    x_multi.append(x)
                    mask_multi.append(resize_mask_like(mask_s, x))
                    if use_attentionmask:
                        exclusion_mask_multi.append(resize_mask_like(ex_mask_s, x))
                        logger.info('exclusionmask shape: ' + str(exclusion_mask_multi[i+1].shape))
                    logger.info('x shape: ' + str(x_multi[i+1].shape))
                    logger.info('mask shape: ' + str(mask_multi[i+1].shape))
                x_multi.reverse()
                mask_multi.reverse()
                if use_attentionmask:
                    exclusion_mask_multi.reverse()
                for i in range(config.LEVELS-1):
                    if use_attentionmask:
                        totalmask = mask_multi[i] + exclusion_mask_multi[i]
                        print('total mask shape:', totalmask.shape)
                    else:
                        totalmask = tf.tile(mask_multi[i], [config.BATCH_SIZE, 1, 1, 1])
                    x, flow = contextual_attention(x, x, totalmask, ksize=config.PATCH_KSIZE, stride=config.PATCH_STRIDE, rate=config.PATCH_RATE)
                    #x, flow = contextual_attention(x, x, mask_multi[i], ksize=3, stride=1, rate=1)
                    flows.append(flow)
                    x = resize(x, scale=2) #TODO: look into using deconv instead of just upsampling
                    x = x * mask_multi[i+1] + x_multi[i+1] * (1.-mask_multi[i+1])
                    logger.info('upsampled x shape: ' + str(x.shape))

            x, offset_flow = contextual_attention(x, x, tf.tile(mask_s, [config.BATCH_SIZE, 1, 1, 1]) if not use_attentionmask else mask_s + ex_mask_s, ksize=config.PATCH_KSIZE, stride=config.PATCH_STRIDE, rate=config.PATCH_RATE)
            flows.append(offset_flow)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10', gating=use_gating)
            pm = x
            x = tf.concat([x_hallu, pm], axis=3) #join branches together

            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11', gating=use_gating)
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12', gating=use_gating)
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample', gating=use_gating)
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14', gating=use_gating)
            x = gen_deconv(x, cnum, name='allconv15_upsample', gating=use_gating)
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16', gating=use_gating)
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, flows

    def build_wgan_local_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_local', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*8, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_global', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*4, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def build_sngan_discriminator(self, x, reuse=False, training=True, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            cnum = 64
            x = sn_dis_conv(x, cnum, name='snconv1', training=training)
            x = sn_dis_conv(x, cnum*2, name='snconv2', training=training)
            x = sn_dis_conv(x, cnum*4, name='snconv3', training=training)
            x = sn_dis_conv(x, cnum*4, name='snconv4', training=training)
            x = sn_dis_conv(x, cnum*4, name='snconv5', training=training)
            x = sn_dis_conv(x, cnum*4, name='snconv6', training=training)
            return x

    def build_graph_with_losses(self, batch_data, config, training=True,
                                summary=False, reuse=False, exclusionmask=None, mask=None):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        use_local_patch = False
        if mask is None:
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
            if config.GAN == 'wgan_gp':
                use_local_patch = True
        else:
            #bbox = (0, 0, config.IMG_SHAPES[0], config.IMG_SHAPES[1])
            mask = tf.cast(tf.less(0.5, mask[:,:,:,0:1]), tf.float32)

        batch_incomplete = batch_pos*(1.-mask)
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training,
            padding=config.PADDING, exclusionmask=exclusionmask)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        l1_alpha = config.COARSE_L1_ALPHA
        # local patches
        if use_local_patch:
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_predicted = local_patch(batch_predicted, bbox)
            local_patch_x1 = local_patch(x1, bbox)
            local_patch_x2 = local_patch(x2, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
        
            losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1)*spatial_discounting_mask(config))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)*spatial_discounting_mask(config))

            losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
            losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        else:
            losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['l1_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2))

        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            if use_local_patch:
                scalar_summary('losses/ae_loss', losses['ae_loss'])
            img_size = [dim for dim in batch_incomplete.shape]
            img_size[2] = 5
            border = tf.zeros(tf.TensorShape(img_size))
            viz_img = [batch_pos, border, batch_incomplete, border, batch_complete, border]
            if not config.PRETRAIN_COARSE_NETWORK:
                batch_complete_coarse = x1*mask + batch_incomplete*(1.-mask)
                viz_img.append(batch_complete_coarse)
                viz_img.append(border)
            if offset_flow is not None:
                scale = 2 << len(offset_flow)
                for flow in offset_flow:
                    viz_img.append(
                        resize(flow, scale=scale,
                               func=tf.image.resize_nearest_neighbor))
                    viz_img.append(border)
                    scale >>= 1
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        # local deterministic patch
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            if not use_local_patch:
                raise Exception('wgan_gp requires global and local patch')
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training=training, reuse=reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse=True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name='g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)
        elif config.GAN == 'sngan':
            if use_local_patch:
                raise Exception('sngan incompatible with global and local patch')
            pos_neg = self.build_sngan_discriminator(batch_pos_neg, name='discriminator', reuse=reuse)
            pos, neg = tf.split(pos_neg, 2)
            g_loss, d_loss = gan_hinge_loss(pos, neg)
            losses['g_loss'] = g_loss
            losses['d_loss'] = d_loss
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss, batch_predicted, name='g_loss')
                scalar_summary('convergence/d_loss', losses['d_loss'])


        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x1, name='g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            gradients_summary(losses['l1_loss'], x1, name='l1_loss_to_x1')
            gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
            if use_local_patch:
                gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
                gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS and use_local_patch:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox=None, name='val', exclusionmask=None):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
            bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name=name+'mask_c')
        batch_pos = batch_data / 127.5 - 1.
        edges = None
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=True,
            training=False, padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        img_size = [dim for dim in batch_incomplete.shape]
        img_size[2] = 5
        border = tf.zeros(tf.TensorShape(img_size))
        viz_img = [border, batch_pos, border, batch_incomplete, border, batch_complete, border]
        if not config.PRETRAIN_COARSE_NETWORK:
            batch_complete_coarse = x1*mask + batch_incomplete*(1.-mask)
            viz_img.append(batch_complete_coarse)
        if offset_flow is not None:
            scale = 2 << len(offset_flow)
            for flow in offset_flow:
                viz_img.append(
                    resize(flow, scale=scale,
                           func=tf.image.resize_nearest_neighbor))
                viz_img.append(border)
                scale >>= 1
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name, exclusionmask=None):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name, exclusionmask=exclusionmask)


    def build_server_graph(self, batch_data, reuse=False, is_training=False, config=None, exclusionmask=None):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=config, exclusionmask=exclusionmask)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete
