import glob
import os
import shutil
from tqdm import tqdm, trange

def main():
    root_path = "../../static_objects/objects/"
    obj_name = "yellow-camero-random" 
    seg_path = os.path.join(root_path, obj_name, 'seg')
    all_views = glob.glob(os.path.join(root_path, obj_name, 'image', "*"))
    out_obj_name = obj_name + '_seg'
    out_path = os.path.join(root_path, out_obj_name, 'image')
    os.makedirs(out_path, exist_ok=True)
    
    for path in tqdm(all_views):
        view = path.split('/')[-1]
        view_image = glob.glob(os.path.join(path, "*.[jp][pn]g"))[0]
        frame_name = view_image.split('/')[-1]
        
        target_dir = os.path.join(out_path, view)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, frame_name.replace('.jpg', '.png'))
        
        src_path = os.path.join(seg_path, frame_name.replace('.jpg', '.png'))
        if os.path.exists(src_path):
            shutil.copyfile(src_path, target_path)
    
    
    

if __name__ == '__main__':
    main()