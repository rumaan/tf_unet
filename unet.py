from tf_unet import unet, util, image_util

data = image_util.ImageDataProvider(
    "extracted_images/scan_*/*.png", data_suffix=u"_flair.png", mask_suffix=u"_mask.png")
print(data)
net = unet.Unet(layers=3, features_root=64, n_class=2, channels=1)
trainer = unet.Trainer(net)
path = trainer.train(data, output_path="checkpoints",
                     training_iters=15, epochs=10)
