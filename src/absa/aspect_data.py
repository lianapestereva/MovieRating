from gettext import translation

label_map = {
    "plot_positive": 0,
    "plot_neutral": 1,
    "plot_negative": 2,
    "acting_positive": 3,
    "acting_neutral": 4,
    "acting_negative": 5,
    "picture_positive": 6,
    "picture_neutral": 7,
    "picture_negative": 8,
    "sound_positive": 9,
    "sound_neutral": 10,
    "sound_negative": 11,
    "humor_positive": 12,
    "humor_neutral": 13,
    "humor_negative": 14
}



translated = {
    "plot":"сюжет",
    "acting":"актерская игра",
    "picture":"визуал",
    "sound":"звук",
    "humor":"юмор"
}


id_to_label = {v: k for k, v in label_map.items()}

label_names = ["O", "B-PLOT", "I-PLOT", "B-ACTING", "I-ACTING", "B-PICTURE", "I-PICTURE", "B-HUMOR", "I-HUMOR",
               "B-SOUND", "I-SOUND"]
tag2id = {tag: i for i, tag in enumerate(label_names)}
id2tag = {i: tag for i, tag in enumerate(label_names)}
