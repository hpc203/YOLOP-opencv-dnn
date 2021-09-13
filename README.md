# YOLOP-opencv-dnn
This repository contained an OpenCV version of YOLOP, a panoptic driving perception network that can handle simultaneously traffic target detection, drivable area segmentation, and lane line detection.

You can find joined to the repository, an onnx file created from the provided weight of YOLOP.

You will find in the repository, a C++ version (main.cpp), a Python version (main.py), an onnx file created from the provided weight of YOLOP and images folder that contains several test images from the bdd100k autopilot dataset.

This program is an opencv inference deployment program based on the recently released [project YOLOP](https://github.com/hustvl/YOLOP) by the vision team of Huazhong University of Science and Technology.
It can be run using only the opencv library, thus completely getting rid of the dependency of any deep learning framework.

This program has been tested with opencv 4.5.3. It doesn't work with opencv 4.2.0 and below.

## Export your own onnx file
You will find in this repository a file export_onnx.py, which is the program that generates the onnx file. If you want to know how to generate .onnx files, you need to copy the export_onnx.py file to the home directory of [YOLOP](https://github.com/hustvl/YOLOP).
You will also need to modify the code in YOLOP/lib/models/common.py as follow :
~~~python
class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain = 2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -&gt; y(b,4c,w/2,h/2)
        # return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        return self.conv(self.contract(x))
~~~
We are adding a Contract class and we have modified the content of the Focus class.
We also need to modify the content of the method forward from the Detect class as follow :
~~~python
    def forward(self, x):
        if not torch.onnx.is_in_onnx_export():
            z = []  # inference output
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                # print(str(i)+str(x[i].shape))
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                # print(str(i)+str(x[i].shape))

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                # print("**")
                # print(y.shape) #[1, 3, w, h, 85]
                # print(self.grid[i].shape) #[1, 3, w, h, 2]
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                """print("**")
                print(y.shape)  # [1, 3, w, h, 85]
                print(y.view(bs, -1, self.no).shape)  # [1, 3*w*h, 85]"""
                z.append(y.view(bs, -1, self.no))
            return x if self.training else (torch.cat(z, 1), x)

        else:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                # print(str(i)&#43;str(x[i].shape))
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                x[i] = torch.sigmoid(x[i])
                x[i] = x[i].view(-1, self.no)
            return torch.cat(x, dim=0)
~~~
After these steps, you can run export_onnx.py to generate the onnx file.
These steps have been extracted from the following Chinese csdn blog post : https://blog.csdn.net/nihate/article/details/112731327

