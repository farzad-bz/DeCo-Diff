from ldm.modules.diffusionmodules.openaimodel import UNetModel


def create_unet_model(latent_size=32, model_channels=256, num_res_blocks=2, num_heads=8, channel_mult=[1,2,4], context_dim=512, ncls=15):
    model = UNetModel(image_size=latent_size, 
                    in_channels=4,
                    model_channels=model_channels, 
                    out_channels=4, 
                    num_heads=num_heads, 
                    num_res_blocks=num_res_blocks, 
                    dropout=0.4,
                    attention_resolutions=[4,2,1], 
                    channel_mult = channel_mult,
                    num_head_channels= model_channels//num_heads,
                    use_spatial_transformer= True,
                    ncls=ncls,
                    transformer_depth= 1,
                    context_dim= context_dim)
    return model


def UNET_XS(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=64, num_heads=4, channel_mult=[1,2,4], context_dim=128, ncls=ncls)


def UNET_S(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=128, num_heads=4, channel_mult=[1,2,4], context_dim=256, ncls=ncls)


def UNET_M(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=192, num_heads=6, channel_mult=[1,2,4], context_dim=384, ncls=ncls)


def UNET_L(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=256, num_heads=8, channel_mult=[1,2,4], context_dim=512, ncls=ncls)


def UNET_XL(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=320, num_heads=12, channel_mult=[1,2,4], context_dim=640, ncls=ncls)




UNET_models = {
'UNet_XS' : UNET_XS, 
'UNet_S' : UNET_S, 
'UNet_M' : UNET_M, 
'UNet_L' : UNET_L, 
'UNet_XL' : UNET_XL, 
}

