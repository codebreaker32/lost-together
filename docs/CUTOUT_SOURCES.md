# Cut-out sources

Segmented with OpenCV `cv2.grabCut` (rectangle init, plus a colour prior for snow/grass on four photos) and morphology clean-up by `lost_together/vision/cutouts.py`. All files below were downloaded from Wikimedia Commons (resized copy served by Commons; original linked). Licences as stated on each Commons file page on 2026-10-26.

| Cut-out | Class | Author / credit | Licence | Source page |
|---|---|---|---|---|
| `bear_black_cdfw` | bear | California Department of Fish and Wildlife from Sacramento, CA, USA | CC BY 2.0 | https://commons.wikimedia.org/wiki/File:Black_Bear_(20261425192).jpg |
| `bear_grizzly_snow` | bear | NPS/Jim Peaco | Public domain | https://commons.wikimedia.org/wiki/File:Grizzly_bear_walking_on_top_of_snow_near_Obsidian_Creek._(d8eed9f2-b700-4f44-9c54-b9d5a0d99409).jpg |
| `bear_grizzly_rocks` | bear | Yellowstone National Park from Yellowstone NP, USA | Public domain | https://commons.wikimedia.org/wiki/File:Grizzly_bear_(8703042570).jpg |
| `person_hiker_desert` | person | Andrew Cattoir (Lake Mead National Recreation Area) | Public domain | https://commons.wikimedia.org/wiki/File:Visitor_hiking_Lake_View_Trail_near_Katherine_Landing_(15c577da-cce7-4057-9c85-9b077a6f2358).jpg |
| `person_hiker_path` | person | Andrew Cattoir (Lake Mead National Recreation Area) | Public domain | https://commons.wikimedia.org/wiki/File:Walking_River_Mountains_Loop_Trail_(a04683af-d98e-46a6-8573-8f5e353b55c4).jpg |
| `deer_doe_standing` | deer | USFWS Midwest Region from Midwest Region, United States | Public domain | https://commons.wikimedia.org/wiki/File:White-tailed_deer_(53485642462).jpg |
| `deer_doe_grazing` | deer | USFWS Midwest Region from Midwest Region, United States | Public domain | https://commons.wikimedia.org/wiki/File:White-tailed_deer_(53485642157).jpg |
| `dog_shepherd` | dog | Hans Kemperman | CC0 | https://commons.wikimedia.org/wiki/File:German-shepherd-4055031920.jpg |

## YOLOX sanity check (cut-out on flat grey, 300 px tall, conf 0.05)

| Cut-out | Top class | Scores |
|---|---|---|
| `bear_black_cdfw` | bear | {'bear': 0.838, 'dog': 0.258} |
| `bear_grizzly_snow` | bear | {'bear': 0.941} |
| `bear_grizzly_rocks` | bear | {'bear': 0.966} |
| `person_hiker_desert` | person | {'person': 0.938, 'remote': 0.145} |
| `person_hiker_path` | person | {'person': 0.912, 'skis': 0.091, 'backpack': 0.055} |
| `deer_doe_standing` | dog | {'dog': 0.926} |
| `deer_doe_grazing` | horse | {'horse': 0.931} |
| `dog_shepherd` | dog | {'dog': 0.952} |

Deer are not a COCO class: the standing doe is reported as `dog`, the grazing doe as `horse`. Neither is reported as `bear` on a neutral background.

Attribution notes:
- `bear_black_cdfw`: CC BY 2.0 - credit "California Department of Fish and Wildlife" (Flickr, via Wikimedia Commons), https://creativecommons.org/licenses/by/2.0/. Modified (segmented, resized, composited).
- NPS / USFWS photos are works of the US federal government (public domain in the US).
- `dog_shepherd`: CC0 (Hans Kemperman, via Pixabay/Wikimedia Commons).
- Backgrounds: TartanGround (Patel et al., arXiv:2505.10696), CC BY 4.0.
