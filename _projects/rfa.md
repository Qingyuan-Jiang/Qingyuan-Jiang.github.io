---
layout: page
title: Human Pose Forecasting
description: Map-Aware Human Pose Prediction for Robot Follow-Ahead
img: assets/img/projects/rfa/thumnail.png
importance: 1
category: work
related_publications: true
---

Check out our project [here](https://qingyuan-jiang.github.io/iros2024_poseForecasting/)

In the robot follow-ahead task, a mobile robot is tasked to maintain its relative position in front of a moving human actor while keeping the actor in sight.
To accomplish this task it is important that the robot understand the full 3D pose of the human (since the head orientation can be different than the torso) and predict the future human poses so as to plan accordingly.
The latter task is tricky in a complex environment with junctions and multiple corridors.
In this work, we address the problem of forecasting the full 3D trajectory of a human in such  environments.
Our main insight is to show that one can first predict the 2D trajectory and then estimate the full 3D trajectory by conditioning the estimator on the predicted 2D trajectory.
With this approach, we achieve results comparable or better than the state-of-the-art three times faster.
As part of our contribution, we present a new dataset where, in contrast to existing datasets, the human motion is in a much larger area than a single room.
We also present a complete robot system that integrates our human pose forecasting network on the mobile robot to enable real-time robot follow-ahead and present results from real-world experiments in multiple buildings on campus.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/rfa/teaser.png" title="teaser image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    We achieve robot follow-ahead task by predicting the human poses with environmental information.
</div>


{% endraw %}
