import torch
import torch.autograd as autograd


EPS = 1e-8

def doppelganger_loss(d_fake,
                      attr_d_fake,
                      d_real,
                      attr_d_real,
                      g_attr_d_coe=1,
                      gradient_penalty=False,
                      discriminator=None,
                      attr_discriminator=None,
                      g_output_feature_train_tf=None,
                      g_output_attribute_train_tf=None,
                      real_feature_pl=None,
                      real_attribute_pl=None,
                      d_gp_coe=0,
                      attr_d_gp_coe=0,
                      ):
    batch_size = d_fake.shape[0]

    g_loss_d = -torch.mean(d_fake)
    g_loss_attr_d = -torch.mean(attr_d_fake)
    g_loss = g_loss_d + g_attr_d_coe * g_loss_attr_d

    if gradient_penalty:
        alpha_dim2 = torch.rand(batch_size, 1)
        alpha_dim3 = torch.unsqueeze(alpha_dim2, dim=2)
        differences_input_feature = g_output_feature_train_tf - real_feature_pl
        interpolates_input_feature = real_feature_pl + alpha_dim3 * differences_input_feature
        differences_input_attribute = g_output_attribute_train_tf - real_attribute_pl
        interpolates_input_attribute =  real_attribute_pl + alpha_dim2 * differences_input_attribute

        interpolates_input_feature.requires_grad = True
        interpolates_input_attribute.requires_grad = True
        disc_interpolates = discriminator(interpolates_input_feature, interpolates_input_attribute)

        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=[interpolates_input_feature, interpolates_input_attribute],
                                  grad_outputs=torch.ones(disc_interpolates.shape),
                                  create_graph=True,
                                  retain_graph=True,
                                  allow_unused=True)
        slopes1 = torch.sum(torch.square(gradients[0]), dim=[1,2])
        slopes2 = torch.sum(torch.square(gradients[1]), dim=1)
        slopes = torch.sqrt(slopes1 + slopes2 + EPS)
        d_loss_gp = ((slopes - 1) ** 2).mean()
    else:
        d_loss_gp = 0

    d_loss_fake = torch.mean(d_fake)
    d_loss_real = -torch.mean(d_real)
    d_loss = d_loss_fake + d_loss_real + d_gp_coe*d_loss_gp

    if gradient_penalty:
        alpha_dim2_attr = torch.rand(batch_size, 1)
        differences_input_attribute = g_output_attribute_train_tf - real_attribute_pl
        interpolates_input_attribute =  real_attribute_pl + alpha_dim2_attr * differences_input_attribute
        interpolates_input_attribute.requires_grad = True
        attr_disc_interpolates = attr_discriminator(interpolates_input_attribute)
        gradients_attr = autograd.grad(outputs=attr_disc_interpolates,
                                       inputs=interpolates_input_attribute,
                                       grad_outputs=torch.ones(attr_disc_interpolates.shape),
                                       create_graph=True,
                                       retain_graph=True,
                                       allow_unused=True)
        slopes1_attr = torch.sum(torch.square(gradients_attr[0]), dim=1)
        slopes_attr = torch.sqrt(slopes1_attr + EPS)
        attr_d_loss_gp = ((slopes_attr - 1) ** 2).mean()
    else:
        attr_d_loss_gp = 0

    attr_d_loss_fake = torch.mean(attr_d_fake)
    attr_d_loss_real = -torch.mean(attr_d_real)
    attr_d_loss = attr_d_loss_fake + attr_d_loss_real + attr_d_gp_coe * attr_d_loss_gp

    return g_loss, d_loss, attr_d_loss
