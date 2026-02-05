|     | import numpy as np                                                     |
|----:|:-----------------------------------------------------------------------|
|   0 | import os                                                              |
|   1 | import sys                                                             |
|   2 | base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) |
|   3 | sys.path.append(base_dir)                                              |
|   4 |                                                                        |
|   5 | def compare(list1, list2):                                             |
|   6 | # sort the list                                                        |
|   7 | list1.sort()                                                           |
|   8 | list2.sort()                                                           |
|   9 | if len(list1) != len(list2):                                           |
|  10 | return False                                                           |
|  11 | for i in range(len(list1)):                                            |
|  12 | if np.isnan(list1[i]):                                                 |
|  13 | if not np.isnan(list2[i]):                                             |
|  14 | return False                                                           |
|  15 | elif list1[i] != list2[i]:                                             |
|  16 | return False                                                           |
|  17 | return True                                                            |
|  18 |                                                                        |
|  19 | def std_digit(list_nums):                                              |
|  20 | new_list = []                                                          |
|  21 | for i in range(len(list_nums)):                                        |
|  22 | new_list.append(round(list_nums[i], 2))                                |
|  23 | return new_list                                                        |
|  24 |                                                                        |
|  25 | def compute_general_chart_metric(references, predictions):             |
|  26 | processed_references = []                                              |
|  27 | processed_predictions = []                                             |
|  28 | for reference in references:                                           |
|  29 | if isinstance(reference, list):                                        |
|  30 | processed_references.extend(reference)                                 |
|  31 | else:                                                                  |
|  32 | processed_references.append(reference)                                 |
|  33 |                                                                        |
|  34 | for prediction in predictions:                                         |
|  35 | if isinstance(prediction, list):                                       |
|  36 | processed_predictions.extend(prediction)                               |
|  37 | else:                                                                  |
|  38 | processed_predictions.append(prediction)                               |
|  39 | processed_references = std_digit(processed_references)                 |
|  40 | processed_predictions = std_digit(processed_predictions)               |
|  41 | return compare(processed_references, processed_predictions)            |
|  42 |                                                                        |
|  43 |                                                                        |
|  44 | def compute_pie_chart_metric(references, predictions):                 |
|  45 | processed_references = []                                              |
|  46 | processed_predictions = []                                             |
|  47 | for reference in references:                                           |
|  48 | if isinstance(reference, list):                                        |
|  49 | processed_references.extend(reference)                                 |
|  50 | else:                                                                  |
|  51 | processed_references.append(reference)                                 |
|  52 | references = processed_references                                      |
|  53 | processed_references = []                                              |
|  54 | total = 0                                                              |
|  55 | for reference in references:                                           |
|  56 | total += reference                                                     |
|  57 | for reference in references:                                           |
|  58 | processed_references.append(round(reference / total, 2))               |
|  59 |                                                                        |
|  60 | for prediction in predictions:                                         |
|  61 | if isinstance(prediction, list):                                       |
|  62 | processed_predictions.extend(prediction)                               |
|  63 | else:                                                                  |
|  64 | processed_predictions.append(prediction)                               |
|  65 | processed_references = std_digit(processed_references)                 |
|  66 | processed_predictions = std_digit(processed_predictions)               |
|  67 | return compare(processed_references, processed_predictions)            |
|  68 |                                                                        |
|  69 |                                                                        |
|  70 | def get_line_y_predictions(plt):                                       |
|  71 | line_y_predctions = []                                                 |
|  72 | lines = plt.gca().get_lines()                                          |
|  73 | line_y_predctions = [list(line.get_ydata()) for line in lines]         |
|  74 | return line_y_predctions                                               |
|  75 |                                                                        |
|  76 |                                                                        |
|  77 | def get_bar_y_predictions(plt):                                        |
|  78 | bar_y_predctions = []                                                  |
|  79 | patches = plt.gca().patches                                            |
|  80 | bar_y_predctions = [patch.get_height() for patch in patches]           |
|  81 | return bar_y_predctions                                                |
|  82 |                                                                        |
|  83 |                                                                        |
|  84 | def get_hbar_y_predictions(plt):                                       |
|  85 | hbar_y_predctions = []                                                 |
|  86 | patches = plt.gca().patches                                            |
|  87 | hbar_y_predctions = [patch.get_width() for patch in patches]           |
|  88 | return hbar_y_predctions                                               |
|  89 |                                                                        |
|  90 |                                                                        |
|  91 | def get_pie_y_predictions(plt):                                        |
|  92 | pie_y_predctions = []                                                  |
|  93 | patches = plt.gca().patches                                            |
|  94 | for patch in patches:                                                  |
|  95 | theta1, theta2 = patch.theta1, patch.theta2                            |
|  96 | value = round((theta2 - theta1) / 360.0, 2)                            |
|  97 | pie_y_predctions.append(value)                                         |
|  98 | return pie_y_predctions                                                |
|  99 |                                                                        |
| 100 |                                                                        |
| 101 | def get_area_y_predictions(plt):                                       |
| 102 | area_y_predctions = []                                                 |
| 103 | area_collections = plt.gca().collections                               |
| 104 | for area_collection in area_collections:                               |
| 105 | area_items = []                                                        |
| 106 | for item in area_collection.get_paths()[0].vertices[:, 1]:             |
| 107 | if item != 0:                                                          |
| 108 | area_items.append(item)                                                |
| 109 | area_y_predctions.append(area_items)                                   |
| 110 | return list(area_y_predctions)                                         |
| 111 |                                                                        |
| 112 |                                                                        |
| 113 | def get_radar_y_predictions(plt):                                      |
| 114 | radar_y_predctions = []                                                |
| 115 | radar_lines = plt.gca().get_lines()                                    |
| 116 | radar_y_predctions = [list(line.get_ydata()) for line in radar_lines]  |
| 117 | for i in range(len(radar_y_predctions)):                               |
| 118 | radar_y_predctions[i] = radar_y_predctions[i][:-1]                     |
| 119 | return radar_y_predctions                                              |
| 120 |                                                                        |
| 121 |                                                                        |
| 122 | def get_scatter_y_predictions(plt):                                    |
| 123 | scatter_y_predctions = []                                              |
| 124 | scatter_collections = plt.gca().collections                            |
| 125 | for scatter_collection in scatter_collections:                         |
| 126 | scatter_items = []                                                     |
| 127 | for item in scatter_collection.get_offsets():                          |
| 128 | scatter_items.append(item[1])                                          |
| 129 | scatter_y_predctions.append(scatter_items)                             |
| 130 | return scatter_y_predctions                                            |
| 131 |                                                                        |
| 132 |                                                                        |
| 133 | def get_waterfall_y_predictions(plt):                                  |
| 134 | waterfall_y_predctions = []                                            |
| 135 | patches = plt.gca().patches                                            |
| 136 | waterfall_y_predctions = [patch.get_height() for patch in patches]     |
| 137 | return waterfall_y_predctions                                          |