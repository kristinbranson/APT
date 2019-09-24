# install boto3 and set up aws cli
import boto3

description = 'AMI for Bransonlab APT backend'
source_ami_id = 'ami-0f31a10294d11388a'
source_region = 'us-east-1'
name = 'bransonlab_apt_ami_tf113_20190419'
owner_id = '535416095125'

##
client = boto3.client('ec2')
regions = client.describe_regions()['Regions']

filters = [
    {'Name': 'name', 'Values': [name, ]},
    {'Name': 'owner-id', 'Values': [owner_id, ]}
]

for reg in regions:
    cur_reg_name = reg['RegionName']
    ec2 = boto3.client('ec2', region_name=cur_reg_name)
    res = ec2.describe_images(Filters=filters)
    if len(res['Images']) > 0:
        print('Image already exists in {}. Skipping'.format(cur_reg_name))
        if len(res['Images'])>1:
            print('Multiple images exist in {}!!!!. Keep only 1'.format(cur_reg_name))
        continue

    print('Image doesnt exist in {}. Copying'.format(cur_reg_name))

    # copy the ami
    client = boto3.client('ec2',region_name=cur_reg_name)
    client.copy_image(Description=description,Encrypted=False,Name=name,
                      SourceImageId=source_ami_id,SourceRegion=source_region)


    ## check the ami
for reg in regions:
    cur_reg_name = reg['RegionName']
    ec2 = boto3.client('ec2', region_name=cur_reg_name)
    res = ec2.describe_images(Filters=filters)
    if len(res['Images']) > 0:
        print('Image already exists in {}. Skipping'.format(cur_reg_name))
    else:
        print('Image still doesnt exist in {}. Check it'.format(cur_reg_name))
