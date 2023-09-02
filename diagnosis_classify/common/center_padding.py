import imageio

def _center_padding(img):
    height,width,_=img.shape
    if width>height:
        img=np.pad(img, (((width-height)//2,width-height-(width-height)//2), (0,0),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0)))
    else:
        img=np.pad(img, ((0,0), ((height-width)//2,height-width-(height-width)//2),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0)))
    # print(img.shape)
    return img
def _read_image_with_protocol(info):
    if info["protocol"] == "DIR":
        path = os.path.join(info["volume"], info["key"])
        try:  # Lots of accidents during image reading, so catch it.
            image_data = imageio.imread(path)
            image_data = _center_padding(image_data)
            imageio.imwrite(path)
        except Exception as e:
            print(info)
            raise
    else:
        raise NotImplementedError

    return image_data

def get_raw_data(self, idx):
    info = self.info[idx]

    if info["image_format"] == "image":
        image = _read_image_with_protocol(info["image_path"])
        assert image.dtype == np.uint8
    else:
        raise NotImplementedError
    assert image.ndim == 3, str(image.shape)
    if image.shape[2] == 4:
        # RGBA -> RGB
        image = image[:, :, :3]
    assert image.shape[2] == 3, "%s has shape: %s" % (info["image_path"], str(image.shape))
    return image.shape
def main():
	info = load_dataset_info_jsonl(dataset_info_paths, "train")
	num_classes = len(label_list)
	LABEL_LIST = [i.strip() for i in open("data/label.list")]
	label2id = {key: i for i, key in enumerate(label_list)}
	for idx in range(len(train_dataset)):
		get_raw_data(idx)
	


if __name__ == '__main__':
	main()

