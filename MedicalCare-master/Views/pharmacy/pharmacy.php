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
  <h1 class="title-block text-center mt-5">
    Nhà thuốc trực tuyến thân thiện dành cho gia đình
  </h1>
  <p class="text-center my-5 h5">
    Mọi nhu cầu chăm sóc sức khoẻ của gia đình bạn sẽ được đáp ứng nhanh
    chóng, tận tâm, và hoàn toàn trực tuyến.
  </p>
  <div class="separator-purple"></div>
  <div class="row mt-3 justify-content-center distance-sm">
    <?php foreach ($categories as $category) : ?>
      <div class="col-xs-6 col-md-4 col-lg-3 col-sm-5 col-6 mt-3 mx-4">
        <a href="?act=product-list&cat_id=<?php echo $category['CategoryID']; ?>" class="a-overlay">
          <div class="card" style="
                background-image: url(./public/img/images/category/<?php echo $category['CategoryImage'] ?>);
                opacity: 1;
                transition-delay: 0.1s;
              ">
            <div class="card-img-overlay">
              <img src="./public/img/images/icons/<?php echo $category['CategoryIcon'] ?>" alt="icon" class="icon-img" />
              <h5 class="card-title text-center font-bold text-white h4">
                <?php echo $category['CategoryName']; ?>
              </h5>
            </div>
          </div>
        </a>
      </div>
    <?php endforeach; ?>
  </div>
</div>
<div class="">
  <img src="./public/img/images/bg/bg-wave.svg" alt="" class="bg" style="position: absolute; z-index: -1; width: 100%;" />


  <div class="container">
    <h2 class="title-block mt-10 text-center my-5">Sản phẩm nổi bật</h2>
    <div class="separator-purple"></div>
    <div class="product-carousel">
      <div class="swiper-wrapper list-card-product">
        <!-- slide -->
        <?php foreach ($top_ratings as $product) : ?>
          <div class="card-slide" style="width: 210px; margin: 0 15px" role="group" aria-label="2 / 17">
            <div class="card" style="height: 500px; max-width: 300px; margin: 0 auto">
              <div class="border-bottom card-img-top rounded-start">
                <img class="" src="./public/img/images/drug/<?php echo $product['Image'] ?>" style="width: 300px; height: 300px; object-fit: cover;" />
              </div>
              <div class="card-body">
                <h5 class="card-title mt-3">
                  <?php echo $product['Name'] ?>
                </h5>
                <h6 class="card-subtitle mt-2 mb-2 text-muted"><?php echo $product['TitleDescription'] ?></h6>

                <div class="mt-10 mb10 card-price text-primary" style="height: 60px">
                  <h6>
                    Giá :
                    <span style="font-size: x-large"><?php echo $product['Price'] ?></span>
                    đ
                  </h6>
                </div>
                <a class="btn button-blue btn-card text-center" href="?act=product-detail&drug_id=<?php echo $product['DrugID']; ?>">
                  Xem sản phẩm
                </a>
                <div class="icons-list">
                  <div class="icon">
                    <img src="./public/img/images/icons/free-shipping.png" alt="" />
                  </div>
                  <div class="icon">
                    <img src="./public/img/images/icons/save-money.png" alt="" />
                  </div>
                  <div class="icon">
                    <img src="./public/img/images/icons/recycle.png" alt="" />
                  </div>
                  <div class="icon">
                    <img src="./public/img/images/icons/customer-service.png" alt="" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        <?php endforeach; ?>
      </div>
    </div>
    <img src="./public/img/images/bg/multi-clinic-bg-2.svg" class="bg" style="position: absolute; height: 400px;left: -10%; z-index: -1;">
    <!-- Mua thuốc theo toa -->
    <h2 class="text-center font-weight-bold title my-5">
      Mua Thuốc Theo Toa Thật Dễ Dàng, An Toàn
    </h2>
    <div class="separator-purple"></div>
    <div id="third" class="container" style="
          display: flex;
          justify-content: space-evenly;
          flex-wrap: wrap;
          gap: 20px;
        ">
      <div class="card" style="
            width: 14rem;
            padding: 20px;
            position: relative;
            min-height: 300px;
          ">
        <img src="https://cdn.jiohealth.com/pharmacy/otc-website/assets/icons/prescription-pic-image@3x.png" style="
              height: 66px;
              width: 40px;
              opacity: 1;
              transition-delay: 0.1s;
              margin: 10px auto;
            " />
        <h4 class="font-weight-bold card-title">
          Chụp Hình Đơn Thuốc Của Bạn
        </h4>
        <p class="text-justify">Và tải ảnh lên website hoặc ứng dụng</p>
      </div>
      <div class="card" style="
            width: 14rem;
            padding: 20px;
            position: relative;
            min-height: 300px;
          ">
        <img src="https://cdn.jiohealth.com/pharmacy/otc-website/assets/icons/pharm-consultation@3x.png" style="
              height: 66px;
              width: 66px;
              opacity: 1;
              transition-delay: 0.1s;
              margin: 10px auto;
            " />
        <h4 class="font-weight-bold card-title">Tham Vấn Dược Sĩ</h4>
        <p class="text-justify">
          Các dược sĩ kinh nghiệm sẽ gọi điện tư vấn và xác nhận đơn thuốc.
          Dược sĩ sẽ tận tình hướng dẫn và trả lời bất cứ câu hỏi nào về việc
          dùng thuốc của bạn.
        </p>
      </div>
      <div class="card" style="
            width: 14rem;
            padding: 20px;
            position: relative;
            min-height: 300px;
          ">
        <img src="https://cdn.jiohealth.com/pharmacy/otc-website/assets/icons/package-icon@3x.png" style="
              height: 45px;
              width: 60px;
              margin-top: 20px;
              opacity: 1;
              transition-delay: 0.1s;
              margin: 10px auto;
            " />
        <h4 class="font-weight-bold card-title">
          Giao Hàng Miễn Phí Trong Hai Giờ
        </h4>
        <p class="text-justify">
          Chúng tôi sẽ cẩn thận đóng gói đơn hàng và giao tận tay vào khung
          giờ thuận tiện nhất cho bạn.
        </p>
      </div>
      <div class="card" style="
            width: 14rem;
            padding: 20px;
            position: relative;
            min-height: 300px;
          ">
        <img src="https://cdn.jiohealth.com/pharmacy/otc-website/assets/icons/card-header-icon@3x.png" style="
              height: 66px;
              width: 72px;
              opacity: 1;
              transition-delay: 0.1s;
              margin: 10px auto;
            " />
        <h4 class="font-weight-bold card-title">Thanh Toán Tiện Lợi</h4>
        <p class="text-justify">
          Bạn có thể thanh toán bằng tiền mặt hoặc thẻ thông qua máy POS khi
          nhận hàng.Bạn cũng có thể yêu cầu xuất hoá đơn VAT nếu cần.
        </p>
      </div>
      <div class="card" style="
            width: 14rem;
            padding: 20px;
            position: relative;
            min-height: 300px;
          ">
        <img src="https://cdn.jiohealth.com/pharmacy/otc-website/assets/icons/contact-icon@3x.png" style="
              height: 66px;
              width: 66px;
              opacity: 1;
              transition-delay: 0.1s;
              margin: 10px auto;
            " />
        <h4 class="font-weight-bold card-title">Hỗ Trợ Tận Tình</h4>
        <p class="text-justify">
          Các dược sĩ sẽ luôn tận tình trả lời những thắc mắc về toa thuốc của
          bạn, từ 8:00 đến 22:00 tất cả các ngày trong tuần.
        </p>
      </div>
    </div>
    <div class="text-center mb-5" style="padding: 20px">
      <div class="btn btn-primary btn-lg" id="create-button">
        Tạo Đơn Thuốc Ngay
      </div>
    </div>
  </div>
</div>