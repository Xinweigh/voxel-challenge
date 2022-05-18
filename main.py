from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(-0.75, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
# scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))

@ti.func
def sphere(pos, r, mat):
    for I in ti.grouped(ti.ndrange((pos[0]-r, pos[0]+r), (pos[1]-r, pos[1]+r), (pos[2]-r, pos[2]+r))):
        sig = False
        if r*(r-1.3) <= dot(I - pos, I - pos) <= r * r:
            sig |= 99 <= dot(I - pos+r, I - pos+r) <= 120
            sig |= 99 <= dot(I - pos-r, I - pos-r) <= 120
            sig |= 450 <= dot(I - (pos+2*r)*0.56, I - (pos+2*r)*0.56) <= 700
            if sig:
                scene.set_voxel(I, mat, (vec3(71.4, 91, 100)+ti.randn())/256)
            else:
                scene.set_voxel(I, mat, (vec3(16.1, 31.0, 62.4)+ti.randn())/256)

@ti.func
def random_pyramid(pos, size, h, mat):
    for I in ti.grouped(ti.ndrange((pos[0]-size, pos[0]+size), 
                        (pos[1]-h,pos[1]+h), (pos[2]-size, pos[2]+size))):
        ratio = size - abs(I.xz - pos.xz)
        if abs(I.y-pos[1]) < h * ratio[0]/size and abs(I.y-pos[1]) < h * ratio[1]/size: 
            scene.set_voxel(I, mat, (vec3(46.7, 64.5, 84.3))/256)

@ti.func
def ground(center, size, r, mat, color, color_noise):
    for I in ti.grouped(
            ti.ndrange((center[0]-r, center[0] + size[0]), (center[1], center[1] + size[1]),
                       (center[2]-r, center[2] + size[2]))):
        if dot(I-center, I-center) < r * r:
            scene.set_voxel(I, mat, color + color_noise * ti.random())

@ti.kernel
def initialize_voxels():
    sphere(ivec3(0, 12, 0), 12, 2)
    for i in range(6):
        random_pyramid(ivec3(12*(i-3)+10*ti.randn(), -24+12*ti.randn(), 12*(i-3)+10*ti.randn()), 3, 10, 2)
    ground(ivec3(0, -48, 0), ivec3(120, 1, 120), 63, 2, vec3(68.6, 86.7, 100.0) / 256, vec3(0.01))

initialize_voxels()

scene.finish()
