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
<?php if ($drugs != null) { ?>
  <div class="container">
    <div class="container-fluid mb-5">
      <div class="row row-cols-1 row-cols-md-4 g-4 mt-3">
        <?php foreach ($drugs as $drug) : ?>
          <div class="col text-center">
            <div class="card" style="width: 300px; height: 500px;">
              <img class="border-bottom card-img-top img-fluid" style="object-fit: fill; width: 300px; height: 300px;" src="./public/img/images/drug/<?php echo $drug['Image'] ?>" alt="drug image">
              <div class="card-body">
                <div class="card-title h5 text-center"><?php echo $drug['Name'] ?></div>
                <p class="card-text"><?php echo $drug['TitleDescription'] ?></p>
              </div>
              <div class="card-footer text-center">
                <form action="?act=product-detail&process=add" method="post">
                  <input type="hidden" name="price" value="<?php echo $drug['Price']; ?>">
                  <input type="hidden" name="image" value="<?php echo $drug['Image']; ?>">
                  <input type="hidden" name="name" value="<?php echo $drug['Name']; ?>">
                  <input type="hidden" name="id" value="<?php echo $drug['DrugID']; ?>">
                  <input name="quantity" type="hidden" class="in-num" value="1">
                  <a class="btn btn-info" href="?act=product-detail&drug_id=<?php echo $drug['DrugID'] ?>">Xem chi tiết</a>
                  <button class="btn btn-primary" type="submit">Thêm vào giỏ</button>
                </form>
              </div>
            </div>
          </div>
        <?php endforeach; ?>
      </div>
      <div class="row mt-5">
        <div class="col-sm-12">
          <div class="pagnation-ul">
            <ul class="clearfix">
              <?php if ($all_drug > 6) {
                for ($i = 1; $i <= ceil($all_drug / 8); $i++) { ?>
                  <li><a href="?act=product-list&cat_id=<?php echo $cat_id ?>&page=<?= $i ?>">
                      <?= $i ?>
                    </a></li>
              <?php }
              }
              ?>
            </ul>
          </div>
        </div>
      </div>
    </div>
  <?php } else {
  require_once('./views/error-404.php');
} ?>
  </div>