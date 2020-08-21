import pyglet
from pyglet.window import key
import ratcave as rc
from vpython import *


# Create Window
window = pyglet.window.Window()
keys = key.KeyStateHandler()
window.push_handlers(keys)

def update(dt):
    pass

pyglet.clock.schedule(update)

# Insert filename into WavefrontReader.
obj_filename = 'model.obj'
obj_reader = rc.WavefrontReader(obj_filename)

# Create Mesh
meshes = [obj_reader.get_mesh(x) for x in obj_reader.bodies]

for m in meshes:
    m.position.xyz = 0, 0, -2


# Create Scene
scene = rc.Scene(meshes=meshes)
scene.bgColor = 0, 0, 0


# Functions to Run in Event Loop
def rotate_meshes(dt):
    for m in meshes:
        m.rotation.y += 15 * dt

pyglet.clock.schedule(rotate_meshes)

@window.event
def on_draw():
    with rc.default_shader:
        scene.draw()


pyglet.app.run()