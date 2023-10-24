import json
import numpy as np

anno="/project/6003167/slee67/bodycam/InternVideo/Downstream/Temporal-Action-Localization/data/InternVideo_Police_BWC_Car_v3.json"

# Opening JSON file
with open(anno) as json_file:
    data = json.load(json_file)
    # data2 = np.array(list(data.items()))

    # Print the type of data variable
    # print("Type:", type(data))

    # Print the data of dictionary
    # print("\ndatabase:", data['database'])
    count=0
    for item in data['database']:
        if data['database'][item]['annotations']:
            for annotation in data['database'][item]['annotations']:
                # print(annotation)
                print("\""+item.replace(' _','_').replace('_ ','_').replace('  ','_').replace(' ','_')+"\"", annotation['segment (frames)'], annotation['label_id'])
            # print("\n\n")
            count+=1
        else:
            print("no annotations")
    # print(count)
    # print(type(data2[2][1]))

