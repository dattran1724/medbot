<div class="tintuc container">
  <div class="wave-bg" style="margin-bottom: -108px;">
    <img src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/images/doctor-profile/elite-care-bg.svg" width="100%"/>
  </div>
  <div class="back-ground ">
    <h1 class="text-center my-5">Tin tức</h1>
    <div class="separator-purple"></div>
    <div class="row mb-5 mx-5 my-5">
      <div id="carouselExampleCaptions" class="carousel slide text-center my-5">
        <div class="carousel-indicators">
          <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
          <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="1" aria-label="Slide 2"></button>
          <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="2" aria-label="Slide 3"></button>
        </div>
        <div class="carousel-inner rounded">
          <?php foreach ($first3News as $key => $new) { ?>
            <?php if ($key == 1) { ?>
              <div class="carousel-item active">
                <img src="./public/img/images/news/<?php echo $new['image'] ?>" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                  <h5 class="bg-dark"><?php echo $new['title'] ?></h5>
                </div>
              </div>
            <?php } else { ?>
              <div class="carousel-item">
                <img src="./public/img/images/news/<?php echo $new['image'] ?>" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                  <h5 class="bg-dark"><?php echo $new['title'] ?></h5>
                </div>
              </div>
          <?php }
          } ?>
        </div>
        <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="prev">
          <span class="carousel-control-prev-icon" aria-hidden="true"></span>
          <span class="visually-hidden">Previous</span>
        </button>
        <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="next">
          <span class="carousel-control-next-icon" aria-hidden="true"></span>
          <span class="visually-hidden">Next</span>
        </button>
      </div>
      <div class="container">
        <?php foreach ($news as $new) : ?>
          <div class="card mb-3 my-4" style="max-height: 500px;">
            <div class="row g-0">
              <div class="col-md-4">
                <img src="./public/img/images/news/<?php echo $new['image'] ?>" class="img-fluid rounded-start" alt="image">
              </div>
              <div class="col-md-8">
                <div class="card-body">
                  <h5 class="card-title text-uppercase"><?php echo $new['title']; ?></h5>
                  <?php $truncatedContent = substr($new['content'], 0, 100) . '...'; ?>
                  <p class="card-text"><?php echo $truncatedContent; ?></p>
                  <p class="card-text"><small class="text-body-secondary"><?php echo $new['created_at']; ?></small></p>
                  <a class="btn card-btn btn-primary" href="?act=news_detail&id=<?php echo $new['id'] ?>">View</a>
                </div>
              </div>
            </div>
          </div>
        <?php endforeach; ?>
      </div>
    </div>
  </div>
</div>