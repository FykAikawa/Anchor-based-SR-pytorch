# Anchor-based-SR-pytorch
**unofficial pytorch implementation of [Anchor-based Plain Net for Mobile Image Super-Resolution](https://arxiv.org/abs/2105.09750)**
scale rate can be changed by changing self.scale of Base7 class.
the number of channels of input images and output images can be changed by changing self.in_channel and self.out_channel.

**How to use**
simply make an instance of Basee7 class by calling Base7().to("cuda:0")
