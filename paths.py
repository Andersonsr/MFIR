import os

root = os.path.dirname(os.path.abspath(__file__))

Paths = {
    'root': root,
    'images': os.path.join(root, 'images/poster_downloads/'),
    'textures': os.path.join(root, 'paper-textures'),
    'masks': os.path.join(root, 'masks'),
    'output': os.path.join(root, 'dataset')
}


