# Generated by Django 2.2.7 on 2019-11-20 07:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('functions', '0012_auto_20191118_1316'),
    ]

    operations = [
        migrations.AlterField(
            model_name='exam',
            name='ansKeyImg',
            field=models.FileField(blank=True, default='default.jpg', null=True, upload_to='exams/answer_key_img/'),
        ),
    ]
