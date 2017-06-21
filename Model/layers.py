from torch import nn
import math

def tf_same_padding(in_dim, kernel_size, strides):
    in_height = in_dim[0]
    in_width = in_dim[1]
    out_height = math.ceil(float(in_height) / float(strides[0]))
    out_width  = math.ceil(float(in_width) / float(strides[1]))
    pad_along_height = max((out_height - 1) * strides[0] +
                           kernel_size[0] - in_height, 0)
    pad_along_width = max((out_width - 1) * strides[1] +
                          kernel_size[1] - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

class custom_con2d(nn.Module):
    def __init__(self, img_size, channel_in, channel_out, kernel, stride=(2, 2)):
        super(custom_con2d, self).__init__()
        self.Conv2d = nn.Conv2d(channel_in, channel_out, kernel, stride=stride)
        self.pad = tf_same_padding(img_size, kernel, stride)

    def forward(self, input):
        out = self.pad(input)
        out = self.Conv2d(out)
        return out