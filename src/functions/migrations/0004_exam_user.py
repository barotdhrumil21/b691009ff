# Generated by Django 2.1.5 on 2019-10-30 07:23

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('functions', '0003_exam'),
    ]

    operations = [
        migrations.AddField(
            model_name='exam',
            name='user',
            field=models.ForeignKey(default='Anonymous', on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
    ]
