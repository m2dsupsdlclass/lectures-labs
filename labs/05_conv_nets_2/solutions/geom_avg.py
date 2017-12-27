heatmap_1_r = imresize(heatmap_1, (50,80)).astype("float32")
heatmap_2_r = imresize(heatmap_2, (50,80)).astype("float32")
heatmap_3_r = imresize(heatmap_3, (50,80)).astype("float32")

heatmap_geom_avg = np.power(heatmap_1_r * heatmap_2_r * heatmap_3_r, 0.333)
display_img_and_heatmap("dog.jpg", heatmap_geom_avg)
