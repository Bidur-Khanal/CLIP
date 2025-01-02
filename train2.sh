# python mismatch_det_from_features.py --feature_mode concat --include_list Attributes Numbers Entities Relations &&
# python mismatch_det_from_features.py --feature_mode concat --include_list Attributes &&
python mismatch_det_from_features.py --feature_mode concat --include_list Numbers &
python mismatch_det_from_features.py --feature_mode concat --include_list Entities &&
python mismatch_det_from_features.py --feature_mode concat --include_list Relations &&
python mismatch_det_from_features.py --feature_mode channel_concat --include_list Attributes Numbers Entities Relations &&
python mismatch_det_from_features.py --feature_mode channel_concat --include_list Attributes &&
python mismatch_det_from_features.py --feature_mode channel_concat --include_list Numbers &&
python mismatch_det_from_features.py --feature_mode channel_concat --include_list Entities &&
python mismatch_det_from_features.py --feature_mode channel_concat --include_list Relations
