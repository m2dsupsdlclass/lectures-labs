from skimage.transform import resize

heatmap_1_r = resize(heatmap_1, (50,80), mode='reflect',
                     preserve_range=True, anti_aliasing=True)
heatmap_2_r = resize(heatmap_2, (50,80), mode='reflect',
                     preserve_range=True, anti_aliasing=True)
heatmap_3_r = resize(heatmap_3, (50,80), mode='reflect',
                     preserve_range=True, anti_aliasing=True)


heatmap_geom_avg = np.power(heatmap_1_r * heatmap_2_r * heatmap_3_r, 0.333)
display_img_and_heatmap("dog.jpg", heatmap_geom_avg)