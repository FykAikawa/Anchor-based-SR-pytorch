# Anchor-based-SR-pytorch
**unofficial pytorch implementation of [Anchor-based Plain Net for Mobile Image Super-Resolution](https://arxiv.org/abs/2105.09750)** <br>
scale rate can be changed by changing self.scale of Base7 class.<br>
the number of channels of input images and output images can be changed by changing self.in_channel and self.out_channel.<br>

**How to use**<br>
simply make an instance of Base7 class by calling Base7().to("cuda:0")<br>
