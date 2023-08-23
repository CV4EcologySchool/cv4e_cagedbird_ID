
        # def MyPad (img):
        #     c, h, w = img.shape
        #     # a list can also be referred to as a sequence (same for a tuple)
        #     target_size = cfg['image_size']
        #     pad_diff_h = target_size[0] - h 
        #     print ("Print the pad_d_h")
        #     print(pad_diff_h)

        #     pad_diff_w = target_size[1] - w

        #     print ("Print the pad_d_w")
        #     print(pad_diff_w)
            
        #     padding = [0, pad_diff_h, pad_diff_w, 0]
        #     padder = Pad(padding)

        #     # don't include all the defaults: https://pytorch.org/vision/main/generated/torchvision.transforms.Pad.html

        #     # define all 4 sides
        #     # 0 for the left and 0 for the bottom
        #     return padder (img)
        
        # self.transform = Compose([  
        #     # to tensor at the end of the Compose function then it will resize the image as a Pillow object and then transform it. Before we have to put the tensor at the end
        #     Resize(size = cfg['image_size'][0], max_size=cfg['image_size'][0]), # so if it is not 
        #     ToTensor(), # ...and convert them to torch.Tensor first, so we can pad it as a tensor later in the same function
        #     Lambda(MyPad),       # For now, we just resize the images to the same dimensions... Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).                
        # ])
