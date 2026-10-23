import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess_aihub_revision as revision
import prepare_aihub_road as adapter


def polygon(label, x0, y0, x1, y1):
    return {"class": label, "polygon": [x0,y0,x1,y0,x1,y1,x0,y1]}


def annotation():
    return {"information": {"filename":"frame.jpg","resolution":[16,12]}, "annotations": [
        polygon("background",0,0,16,12), polygon("freespace",0,5,16,12),
        polygon("trafficDrum",1,6,4,10), polygon("warningTriangle",6,5,8,8),
        polygon("whiteLane",10,5,12,12), polygon("new_unknown",0,0,2,2),
        polygon("vehicle",3,8,6,12)]}


def test_explicit_obstacles_only_and_unknown_stays_ignored():
    doc = annotation()
    old,_ = adapter.rasterize(doc,(16,12),'frame.jpg')
    new,objects,changed,_ = revision.correct_mask(old,doc,adapter,'frame.jpg')
    assert new[7,2] == new[6,7] == 1
    assert old[7,2] == old[6,7] == 255
    assert new[8,11] == new[1,1] == 255
    assert new[8,14] == 0
    assert new[10,4] == old[10,4] == 1
    assert len(objects)==2 and changed>0
    assert np.all(old[new!=old]==255)
    assert np.all(new[new!=old]==1)


def test_wrong_parent_mask_rejected():
    doc=annotation();old,_=adapter.rasterize(doc,(16,12),'frame.jpg')
    old[0,15]=0
    with pytest.raises(ValueError,match='reproduce'):
        revision.correct_mask(old,doc,adapter,'frame.jpg')


def test_polygon_order_invariant():
    doc=annotation();old,_=adapter.rasterize(doc,(16,12),'frame.jpg')
    a=revision.correct_mask(old,doc,adapter,'frame.jpg')[0]
    doc['annotations'].reverse()
    b=revision.correct_mask(old,doc,adapter,'frame.jpg')[0]
    np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('coords', [[-1,0,2,0,2,2],[0,0,1,1,2,2],[0,0,float('nan'),1,2,2]])
def test_bad_geometry_not_repaired(coords):
    doc=annotation();old,_=adapter.rasterize(doc,(16,12),'frame.jpg')
    doc['annotations'][2]['polygon']=coords
    with pytest.raises(ValueError,match='Invalid original'):
        revision.correct_mask(old,doc,adapter,'frame.jpg')


def fixture_row(tmp_path):
    doc=annotation();old,inventory=adapter.rasterize(doc,(16,12),'frame.jpg')
    im=tmp_path/'frame.jpg';Image.new('RGB',(16,12),'white').save(im)
    target=tmp_path/'mask.png';Image.fromarray(old).save(target)
    ann=tmp_path/'frame.json';ann.write_text(json.dumps(doc),encoding='utf-8')
    row={'split':'train','export_split':'train','dataset_id':'aihub_13','sample_id':'clip/frame.jpg',
         'image':str(im),'image_sha256':revision.sha(im),'image_rgb_sha256':'0'*64,
         'target':str(target),'target_sha256':revision.sha(target),'width':16,'height':12,
         'target_pixel_counts':revision.counts(old),'target_original':str(ann),
         'target_original_sha256':revision.sha(ann),'source_image':str(im),
         'source_class_counts':dict(inventory),'export_id':'fixture',
         'ignored_class_counts':{'trafficDrum':1,'warningTriangle':1,'whiteLane':1,'new_unknown':1}}
    out=tmp_path/'new';(out/'masks'/'aihub_13').mkdir(parents=True)
    return row,out


def test_processing_preserves_original_and_records_pixel_changes(tmp_path):
    row,out=fixture_row(tmp_path);before=dict(row)
    new,change=revision.process_row(row,out,adapter,'new_mapping')
    assert row==before
    assert revision.sha(row['target'])==row['target_sha256']
    assert revision.sha(row['target_original'])==row['target_original_sha256']
    assert new['target']!=row['target']
    assert 'trafficDrum' not in new['ignored_class_counts']
    assert new['ignored_class_counts']['whiteLane']==1
    assert new['target_pixel_counts']['0']==row['target_pixel_counts']['0']
    assert new['target_pixel_counts']['1']-row['target_pixel_counts']['1']==change['ignore_to_nonroad_pixels']


@pytest.mark.parametrize('field',['image_sha256','target_sha256','target_original_sha256'])
def test_modified_source_rejected(tmp_path,field):
    row,out=fixture_row(tmp_path);row[field]='f'*64
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        revision.process_row(row,out,adapter,'new_mapping')
    assert not list((out/'masks'/'aihub_13').glob('*.png'))


def test_official_validation_cannot_be_processed(tmp_path):
    row,out=fixture_row(tmp_path);row['split']='val'
    with pytest.raises(ValueError,match='Only the frozen training'):
        revision.process_row(row,out,adapter,'new_mapping')


@pytest.mark.parametrize('field',['image_sha256','image_rgb_sha256','group_id','origin_id'])
def test_reporting_overlap_rejected(field):
    a={'dataset_id':'x','image_sha256':'a','image_rgb_sha256':'b','group_id':'c','origin_id':'d'}
    b={'dataset_id':'x','image_sha256':'e','image_rgb_sha256':'f','group_id':'g','origin_id':'h'}
    b[field]=a[field]
    with pytest.raises(ValueError,match='overlap'):
        revision.check_protected_overlap([a],[b])


def test_existing_output_never_overwritten(tmp_path):
    policy=tmp_path/'policy.json';policy.write_text('{}')
    with pytest.raises(ValueError,match='Output exists'):
        revision.run(policy,tmp_path)


def test_unknown_mask_values_rejected():
    with pytest.raises(ValueError,match='unsupported'):
        revision.counts(np.array([[0,1,255,2]],dtype=np.uint8))


def test_quarantine_allows_replay_identity_and_rejects_missing_aihub_identity():
    excluded={('aihub_13','bad.jpg')}
    revision.check_quarantine([{'dataset_id':'vkitti2_0_3','frame_id':'1'}],excluded)
    with pytest.raises(ValueError,match='identity required'):
        revision.check_quarantine([{'dataset_id':'aihub_13'}],excluded)
    with pytest.raises(ValueError,match='reintroduced'):
        revision.check_quarantine([{'dataset_id':'aihub_13','sample_id':'bad.jpg'}],excluded)
