from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, load_masks
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import argparse
import torch
import numpy as np
import open3d as o3d
import os
from utils import features_to_world_space_mesh, save_mesh

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=f'cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--schedule', type=str, default=f'cosine')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--edge_threshold', type=float, default=0.48)
    parser.add_argument('--data_path', type=str, default=f'./data')
    parser.add_argument('--save_path', type=str, default=f'./output')
    parser.add_argument('--not_use_mask', action='store_true') 

    args = parser.parse_args()

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(args.device)
    # load_images can take a list of images or a directory
    images = load_images(os.path.join(args.data_path, 'rgb'), size=512)
    if not args.not_use_mask:
        masks = load_masks(os.path.join(args.data_path, 'mask'), size=512, erode_iter=1)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=args.batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    if args.not_use_mask:
        masks = [np.ones_like(img, dtype=bool)[..., 0] for img in imgs]

    points = np.concatenate([p.detach().cpu().numpy()[m] for p, m in zip(pts3d, masks)]).reshape(-1, 3)
    colors = np.concatenate([img[m] for img, m in zip(imgs, masks)]).reshape(-1, 3)

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    pcd.colors = o3d.utility.Vector3dVector(colors) 
    o3d.io.write_point_cloud(os.path.join(args.save_path, f"pcd.ply"), pcd)

    # Create a mesh
    vertices_all = torch.empty((3, 0), device=args.device)
    colors_all = torch.empty((3, 0), device=args.device)
    faces_all = torch.empty((3, 0), device=args.device, dtype=torch.long)

    for i, (pts, img, msk) in enumerate(zip(pts3d, imgs, masks)):
        vertices, faces, colors = features_to_world_space_mesh(
            world_space_points=torch.permute(pts.detach(), (2, 0, 1)),
            colors=torch.permute(torch.tensor(img, device=args.device), (2, 0, 1)),
            mask=torch.tensor(msk, device=args.device),
            edge_threshold=args.edge_threshold
        )

        save_mesh(vertices, faces, colors, os.path.join(args.save_path, f"mesh_{i}.ply"))

        faces += vertices_all.shape[1]  # add face offset

        vertices_all = torch.cat([vertices_all, vertices], dim=1)
        colors_all = torch.cat([colors_all, colors], dim=1)
        faces_all = torch.cat([faces_all, faces], dim=1)
    
    save_mesh(vertices_all, faces_all, colors_all, os.path.join(args.save_path, f"mesh.ply"))
