s = "n02084071"

probas_1 = forward_pass_resize("dog.jpg", (200, 320))
heatmap_1 = build_heatmap(probas_1, synset=s)
display_img_and_heatmap("dog.jpg", heatmap_1)

probas_2 = forward_pass_resize("dog.jpg", (400, 640))
heatmap_2 = build_heatmap(probas_2, synset=s)
display_img_and_heatmap("dog.jpg", heatmap_2)

probas_3 = forward_pass_resize("dog.jpg", (800, 1280))
heatmap_3 = build_heatmap(probas_3, synset=s)
display_img_and_heatmap("dog.jpg", heatmap_3)

# We observe that heatmap_1 and heatmap_2 gave coarser 
# segmentations than heatmap_3. However, heatmap_3
# has small artifacts outside of the dog area
# heatmap_3 encodes more local, texture level information
# about the dog, while lower resolutions will encode more
# semantic information about the full object
# combining them is probably a good idea!