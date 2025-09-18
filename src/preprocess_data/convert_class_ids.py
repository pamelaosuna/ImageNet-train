
import scipy.io

# 0-indexed
# class ID as in https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
CLASS_IDS = [
    1, # goldfish
    2, # great white shark
    6, # stingray
    113, # snail
    151, # chihuahua
    309, # bee
    310, # ant
    315, # mantis
    398, # abacus
    402, # acoustic guitar
    405, # airship
    407, # ambulance
    420, # banjo
    430, # basketball
    462, # broom
    472, # canoe
    479, # car wheel
    486, # violoncello
    496, # christmas stocking
    504, # coffee mug
    508, # computer keyboard
    510, # container ship
    526, # desk
    546, # electric guitar
    574, # golf ball
    596, # hatchet
    604, # hourglass
    650, # microphone
    673, # computer mouse
    721, # pillow
    732, # polaroid
    789, # shoji
    808, # sombrero
    852, # tennis ball
    879, # umbrella
    890, # volleyball
    947, # mushroom
    949, # strawberry
    951, # lemon
    952, # fig
    953, # pineapple
    954, # banana
    955, # jackfruit
    956, # custard apple
    957, # pomegranate
    985, # daisy
    987, # corn
    995, # earthstar
    997, # bolete
    999, # toilet paper
]


if __name__ == "__main__":
    devkit_meta = '/Users/pamelaosuna/Downloads/meta.mat'
    meta = scipy.io.loadmat(devkit_meta, squeeze_me=True)['synsets']

    cls2name_fp = 'imagenet1000_clsidx_to_labels.txt'
    
    # read txt file as dict that labels idx to name
    with open(cls2name_fp, 'r') as f:
        lines = f.read()
        cls2name = eval(lines)
    
    # 1 -> n01443537
    # 2 -> n01484850
    # ...
    # first convert class id to name using cls2, then name to wind
    names = [cls2name[cid] for cid in CLASS_IDS]

    # meta[0] is (1, 'n02119789', 'kit fox, Vulpes macrotis', 'small grey fox of southwestern United States; may be a subspecies of Vulpes velox', 0, array([], dtype=uint8), 0, 1300)
    name2wnid = {m[2]: m[1] for m in meta}
    wnid_labels = [name2wnid[name] for name in names]
    
    # save list of wind labels to txt file
    with open('sub50_imagenet_labels.txt', 'w') as f:
        for wnid in wnid_labels:
            f.write(f"{wnid}\n")


