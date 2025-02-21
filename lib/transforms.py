from __future__ import annotations

from typing import List
from collections.abc import Hashable, Mapping
from monai.config import NdarrayOrTensor
from monai.transforms import RandomizableTransform, MapTransform
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta
import random, torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import cv2, time

def rgb2hsv(im):
    # Source: https://github.com/TheAlgorithms/Python/blob/master/conversions/rgb_hsv_conversion.py
    # im_{i} \in [0,1]

    value = torch.max(im, dim=0)[0]
    chroma = value - torch.min(im, dim=0)[0]
    saturation = 0*(value == 0) + (value!=0)*(chroma / (value+1e-10))

    c1 = (chroma==0)
    v1 = 0
    c2 = (value==im[0])
    v2 = 60 * (0 + (im[1] - im[2]) / (chroma+1e-10))
    c3 = (value==im[1])
    v3 = 60 * (2 + (im[2] - im[0]) / (chroma+1e-10))
    c4 = ~c1 * ~c2 * ~c3
    v4 = 60 * (4 + (im[0] - im[1]) / (chroma+1e-10))

    hue = ((c1*v1 + c2*v2 + c3*v3 + c4*v4) + 360) % 360

    return [hue, saturation, value]

def hsv2rgb(hue, saturation, value):
    chroma = value * saturation
    hue_section = hue / 60
    second_largest_component = chroma * (1 - torch.abs(hue_section % 2 - 1))
    match_value = value - chroma

    c1 = (hue_section >= 0)*(hue_section <= 1)
    red_v1 = chroma + match_value
    green_v1 = second_largest_component + match_value
    blue_v1 = match_value

    c2 = (hue_section > 1)*(hue_section <= 2)
    red_v2 = second_largest_component + match_value
    green_v2 = chroma + match_value
    blue_v2 = match_value

    c3 = (hue_section > 2)*(hue_section <= 3)
    red_v3 = match_value
    green_v3 = chroma + match_value
    blue_v3 = second_largest_component + match_value

    c4 = (hue_section > 3)*(hue_section <= 4)
    red_v4 = match_value
    green_v4 = second_largest_component + match_value
    blue_v4 = chroma + match_value

    c5 = (hue_section > 4)*(hue_section <= 5)
    red_v5 = second_largest_component + match_value
    green_v5 = match_value
    blue_v5 = chroma + match_value

    c6 = ~c1 * ~c2 * ~c3 * ~c4 * ~c5
    red_v6 = chroma + match_value
    green_v6 = match_value
    blue_v6 = second_largest_component + match_value

    red = c1*red_v1 + c2*red_v2 + c3*red_v3 + c4*red_v4 + c5*red_v5 + c6*red_v6
    green = c1*green_v1 + c2*green_v2 + c3*green_v3 + c4*green_v4 + c5*green_v5 + c6*green_v6
    blue = c1*blue_v1 + c2*blue_v2 + c3*blue_v3 + c4*blue_v4 + c5*blue_v5 + c6*blue_v6

    im = torch.stack([red, green, blue])

    return im

class RandHued(RandomizableTransform, MapTransform):
    def __init__(self, keys: List[str], prob: float) -> None:

        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.randhue = RandHue(prob=1.0)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        for key in self.key_iterator(d):
            d[key] = self.randhue(
                    convert_to_tensor(d[key], track_meta=get_track_meta()))
        return d

class RandHue(RandomizableTransform):
    def __init__(self, prob: float) -> None:
        RandomizableTransform.__init__(self, prob)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        orig_min = img.min()
        orig_range = img.max() - img.min()

        img = (img - orig_min) / orig_range
        #print("  c", img.min())
        h, s, v = rgb2hsv(img)
        # Randomize
        r1 = torch.rand(())
        r2 = torch.rand(())
        r3 = torch.rand(())
        h = torch.ones(h.shape) * r1 * 360
        s = torch.clip(s + r2*0.5, 0, 1)
        v = torch.clip(v + r3*0.5, 0, 1)
        img = hsv2rgb(h, s, v)
        #print("  d", img.min())

        img = (img * orig_range) + orig_min

        #print(img.min(), r1, r2, r3, orig_range, orig_min, np.isnan(img.min()))


        return img
