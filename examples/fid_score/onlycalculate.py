from cleanfid import fid

# 定义真实图片和生成图片的路径
real_images_dir = "evaluation/real"
generated_images_dir = "evaluation/generated"

# 计算 FID 分数
score = fid.compute_fid(
    fdir1=real_images_dir,
    fdir2=generated_images_dir,
    mode="clean",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"FID score: {score}")
