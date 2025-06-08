<div class="search-top">
  <div class="search text-center">
    <div class="input-group my-3">
      <span class="input-group-text span-icon" style="height: 100%"><i class="fa-solid fa-magnifying-glass"></i></span>
      <input type="text" class="form-control" name="search" placeholder="Tìm kiếm tên, nhãn hiệu, loại thuốc" />
      <span class="input-group-text span-text"><a href="">Search</a></span>
    </div>
  </div>
  <div style="display: flex;align-items: center;flex-wrap: wrap;justify-content: center;">
    <div class="cart-wrapper me-5 ms-4 my-2">
      <a class="btn" href="?act=product-detail&process=list_order">
        <i class="fa-solid fa-cart-plus"></i>
      </a>
    </div>
  </div>
</div>
<div class="container">
  <div class="wrapper row mt-5">
    <div class="preview col-md-6">
      <img class="img-thumbnail mx-auto" style="max-width: 400px; max-height: 400px; object-fit: contain;" src="./public/img/images/drug/<?php echo $drug['Image']; ?>" alt="<?php echo $drug['Name']; ?>" style="object-fit: contain;">
    </div>
    <div class="details col-md-6">
      <h3 class="product-title"><?php echo $drug['Name'] ?></h3>
      <div class="rating">
        <div class="stars">
          <?php
          $i = 1;
          while ($i <= ceil($avg_rating)) {
          ?>
            <span class="fa fa-star checked"></span>
          <?php $i++;
          }
          while ($i <= 5) { ?>
            <span class="fa fa-star"></span>
          <?php $i++;
          } ?>
        </div>
        <span class="review-no"><?php echo ($rating_drug != null ?  count($rating_drug) : 0) . ($rating_drug != null && count($rating_drug) >= 2 ? " reviews" : " review") ?></span>
      </div>
      <p class="product-description"><?php echo $drug['TitleDescription'] ?></p>
      <h4 class="price">Giá: <span><?php echo $drug['Price'] ?></span> đ</h4>
      <form action="?act=product-detail&process=add" method="post">
        <div class="num-block skin-2 mb-3">
          <div class="num-in">
            <span class="minus dis"></span>
            <input type="hidden" name="price" value="<?php echo $drug['Price']; ?>">
            <input type="hidden" name="image" value="<?php echo $drug['Image']; ?>">
            <input type="hidden" name="name" value="<?php echo $drug['Name']; ?>">
            <input type="hidden" name="id" value="<?php echo $drug['DrugID']; ?>">
            <input name="quantity" type="text" class="in-num" value="1">
            <span class="plus"></span>
          </div>
        </div>
        <div class="action">
          <button type="submit" class="btn btn-info btn-lg text-uppercase">Thêm vào giỏ hàng</button>
        </div>
      </form>
    </div>
  </div>
  <div class="wrapper row mt-5">
    <div class="accordion accordion-flush" id="accordionExample">
      <div class="accordion-item">
        <h2 class="accordion-header">
          <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne<?php echo $drug['DrugID'] ?>" aria-expanded="false" aria-controls="collapseOne<?php echo $drug['DrugID'] ?>">
            <i class="fa-solid fa-circle-plus"></i>
            <div class="text-uppercase h5">Thông tin thuốc</div>
          </button>
        </h2>
        <div id="collapseOne<?php echo $drug['DrugID'] ?>" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
          <div class="accordion-body">
            <p><?php echo $drug['DrugInformation'] ?></p>
          </div>
        </div>
      </div>
      <div class="accordion-item">
        <h2 class="accordion-header">
          <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseTwo<?php echo $drug['DrugID'] ?>" aria-expanded="false" aria-controls="collapseTwo<?php echo $drug['DrugID'] ?>">
            <i class="fa-solid fa-circle-plus"></i>
            <div class="text-uppercase h5">Thành Phần</div>
          </button>
        </h2>
        <div id="collapseTwo<?php echo $drug['DrugID'] ?>" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
          <div class="accordion-body">
            <p><?php echo $drug['Ingredient'] ?></p>
          </div>
        </div>
      </div>
      <div class="accordion-item">
        <h2 class="accordion-header">
          <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseThree<?php echo $drug['DrugID'] ?>" aria-expanded="false" aria-controls="collapseThree<?php echo $drug['DrugID'] ?>">
            <i class="fa-solid fa-circle-plus"></i>
            <div class="text-uppercase h5">Hướng dẫn sử dụng</div>
          </button>
        </h2>
        <div id="collapseThree<?php echo $drug['DrugID'] ?>" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
          <div class="accordion-body">
            <p><?php echo $drug['Instruction'] ?></p>
          </div>
        </div>
      </div>
      <div class="accordion-item">
        <h2 class="accordion-header">
          <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseFour<?php echo $drug['DrugID'] ?>" aria-expanded="false" aria-controls="collapseFour<?php echo $drug['DrugID'] ?>">
            <i class="fa-solid fa-circle-plus"></i>
            <div class="text-uppercase h5">Lưu ý</div>
          </button>
        </h2>
        <div id="collapseFour<?php echo $drug['DrugID'] ?>" class="accordion-collapse collapse" data-bs-parent="#accordionExample">
          <div class="accordion-body">
            <p><?php echo $drug['Attention'] ?></p>
          </div>
        </div>
      </div>
      <hr class="border border-2 opacity-75">
      <div class="wrapper row mt-5">
        <h3 class="text-uppercase mb-4">Đánh giá</h3>
        <?php if (isset($_SESSION['check_rating']) && $_SESSION['check_rating'] == true) : ?>
          <div class="row text-center bg-success">Đánh giá đã được gửi!</div>
        <?php endif; ?>
        <div class="row">
          <div class="col-md-4">
            <h4 class="text-uppercase text-center">điểm trung bình</h4>
            <h1 class="text-center text-warning"><?php echo $avg_rating ?>/5</h1>
          </div>
          <div class="col-md-4">
            <?php if (isset($star_rating) && $star_rating != null) { ?>
              <?php foreach ($star_rating as $star => $count) { ?>
                <div class="row">
                  <div class="col-sm-2"><?php echo $star ?> <span class="fa fa-star checked"></span> </div>
                  <div class="col-sm-10">
                    <div class="progress" role="progressbar" aria-label="Warning example" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                      <div class="progress-bar bg-warning" style="width: <?php echo ($count / count($rating_drug)) * 100 ?>%"><?php echo $count ?></div>
                    </div>
                  </div>
                </div>
              <?php }
            } else {
              for ($i = 1; $i <= 5; $i++) {
              ?>
                <div class="row">
                  <div class="col-sm-2"><?php echo $i ?> <span class="fa fa-star checked"></span></div>
                  <div class="col-sm-10">
                    <div class="progress" role="progressbar" aria-label="Warning example" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                      <div class="progress-bar bg-warning" style="width: 0%"><?php echo 0 ?></div>
                    </div>
                  </div>
                </div>
            <?php }
            } ?>
          </div>
          <div class="col-md-4 my-auto">
            <div class="row">
              <?php
              if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) {
              ?>
                <button type="button" class="btn btn-primary text-uppercase btn-lg" data-bs-toggle="modal" data-bs-target="#exampleModal">
                  Đánh giá sản phẩm này
                </button>
              <?php } else { ?>
                <a href="?act=account&process=login" class="btn btn-info btn-lg text-center text-uppercase">
                  Đăng nhập để đánh giá
                </a>
              <?php } ?>
              <!-- Modal -->
              <div class="modal fade" id="exampleModal" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable">
                  <div class="modal-content">
                    <div class="modal-header">
                      <h1 class="modal-title fs-5" id="exampleModalLabel"><?php echo $drug['Name'] ?></h1>
                      <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                      <form action="?act=product-rating" method="post">
                        <input type="hidden" name="drug_id" value="<?php echo $drug['DrugID'] ?>">
                        <input type="hidden" name="user_id" value="<?php echo $_SESSION['user']['userID'] ?>">
                        <div class="row mb-3">
                          <div class="col-sm-2">
                            <input class="form-check-input" type="radio" name="rating_num" id="inlineRadio1" value="1">
                            <label class="form-check-label" for="inlineRadio1">1</label>
                          </div>
                          <div class="col-sm-2">
                            <input class="form-check-input" type="radio" name="rating_num" id="inlineRadio2" value="2">
                            <label class="form-check-label" for="inlineRadio2">2</label>
                          </div>
                          <div class="col-sm-2">
                            <input class="form-check-input" type="radio" name="rating_num" id="inlineRadio3" value="3">
                            <label class="form-check-label" for="inlineRadio3">3</label>
                          </div>
                          <div class="col-sm-2">
                            <input class="form-check-input" type="radio" name="rating_num" id="inlineRadio4" value="4">
                            <label class="form-check-label" for="inlineRadio4">4</label>
                          </div>
                          <div class="col-sm-2">
                            <input class="form-check-input" type="radio" name="rating_num" id="inlineRadio5" value="5">
                            <label class="form-check-label" for="inlineRadio5">5</label>
                          </div>
                        </div>
                        <div class="row">
                          <div class="mb-3">
                            <label for="exampleFormControlInput1" class="form-label">Tiêu đề</label>
                            <input type="text" name="title" class="form-control" id="exampleFormControlInput1">
                          </div>
                          <div class="mb-3">
                            <label for="exampleFormControlTextarea1" class="form-label">Bình luận</label>
                            <textarea class="form-control" name="comments" id="exampleFormControlTextarea1" rows="3"></textarea>
                          </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                      <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Huỷ</button>
                      <button type="submit" class="btn btn-primary">Đánh giá</button>
                    </div>
                  </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
      <hr class="border border-2 opacity-75">
      <div class="wrapper row mt-5">
        <h3 class="text-uppercase">Bình luận</h3>
      </div>
      <?php if (empty($comment_drug)) { ?>
        <div class="mx-auto my-auto" style="display: inline-block; text-align: center; height: 300px;"></div>
        <?php } else {
        foreach ($comment_drug as $comment) : ?>
          <div class="card mb-5 border">
            <div class="row g-0">
              <div class="col-md-8">
                <div class="card-body">
                  <div class="card-title h2 text-uppercase mb-2"><?php echo $comment['name'] ?></div>
                  <p class="card-text h3"><?php echo $comment['title'] ?> <br> <?php echo $comment['comments'] ?></p>
                  <p class="card-text "><small class="text-muted"> <?php echo $comment['created'] ?></small></p>
                </div>
              </div>
            </div>
          </div>
      <?php endforeach;
      } ?>

    </div>
  </div>
</div>