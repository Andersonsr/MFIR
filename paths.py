import os

root = os.path.dirname(os.path.abspath(__file__))

Paths = {
    'root': root,
    'images': os.path.join(root, 'images/'),
    'textures': os.path.join(root, 'paper-textures'),
    'masks': os.path.join(root, 'masks'),
    'output': os.path.join(root, 'dataset')
}


