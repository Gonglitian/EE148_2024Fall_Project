# all the classes that are present in the dataset
CLASSES = ['animal', 'archway', 'bicyclist', 'bridge', 'building', 'car',
        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve',
        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving',
        'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk',
        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight',
        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
        'wall']

label_colors_list = [
        (64, 128, 64), # animal
        (192, 0, 128), # archway
        (0, 128, 192), # bicyclist
        (0, 128, 64), #bridge
        (128, 0, 0), # building
        (64, 0, 128), #car
        (64, 0, 192), # car luggage pram...???...
        (192, 128, 64), # child
        (192, 192, 128), # column pole
        (64, 64, 128), # fence
        (128, 0, 192), # lane marking driving
        (192, 0, 64), # lane maring non driving
        (128, 128, 64), # misc text
        (192, 0, 192), # motor cycle scooter
        (128, 64, 64), # other moving
        (64, 192, 128), # parking block
        (64, 64, 0), # pedestrian
        (128, 64, 128), # road
        (128, 128, 192), # road shoulder
        (0, 0, 192), # sidewalk
        (192, 128, 128), # sign symbol
        (128, 128, 128), # sky
        (64, 128, 192), # suv pickup truck
        (0, 0, 64), # traffic cone
        (0, 64, 64), # traffic light
        (192, 64, 128), # train
        (128, 128, 0), # tree
        (192, 128, 192), # truck/bus
        (64, 0, 64), # tunnel
        (192, 192, 0), # vegetation misc.
        (0, 0, 0),  # 0=background/void
        (64, 192, 0), # wall
    ]