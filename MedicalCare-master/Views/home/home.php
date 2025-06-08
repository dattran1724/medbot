<?php
require_once("banner.php")
    ?>

<div class="chuyenkhoa">
    <div class="container">
        <h1 class="my-5">Chuyên khoa phổ biến</h1>
        <div class="separator-purple"></div>
        <div class="chuyenkhoa-slide">
            <?php
            if (isset($data_ck) and $data_ck != NULL) {
                foreach ($data_ck as $value) {
                    ?>
                    <div class="khoa-item" style="margin:10px;max-width:220px" >
                        <div style="background-color: #eee;border-radius: 30px;padding: 5px;">
                            <a href="?act=chitietchuyenkhoa&id=<?= $value['id_ck'] ?>" class = "div-icon m-4">
                                <img src="./public/img/images/specialist/<?php echo $value['hinhanh'] ?>"
                                    alt="<?php echo $value['hinhanh'] ?>" />
                            </a>
                        </div>

                        <a href="?act=chitietchuyenkhoa&id=<?= $value['id_ck'] ?>">
                            <h4 style="text-align: center;">
                                <?php echo $value['name'] ?>
                            </h4>
                        </a>

                    </div>

                <?php }
            } else {
                echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
            }
            ?>
        </div>
    </div>
</div>
<div class="giaiphap">
    <img src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/images/for-press/health-resources-bg.svg
                " alt="" class="bg" />
    <div class="container">
        <h1 class="my-5">Các bước dễ dàng cho giải pháp của bạn</h1>
        <div class="separator-purple"></div>
        <div class="row list-card">
            <div class="card my-3" style="width: 19rem">
                <img src="./public/img/giaiphap1.png" class="card-img-top" alt="..." />
                <div class="card-body">
                    <div class="card-text">
                        <h3 style="font-size: 22px">Tìm bác sĩ</h3>
                        <p style="text-align: justify">
                            Nghiên cứu và tìm hiểu về các bác sĩ hoặc chuyên gia trong
                            lĩnh vực liên quan đến vấn đề sức khỏe của bạn.
                        </p>
                    </div>
                </div>
            </div>
            <div class="card my-3" style="width: 19rem">
                <img src="./public/img/giaiphap2.png" class="card-img-top" alt="..." />
                <div class="card-body">
                    <div class="card-text">
                        <h3 style="font-size: 22px">Yêu cầu tư vấn</h3>
                        <p style="text-align: justify">
                            Liên hệ với bác sĩ hoặc phòng khám để yêu cầu tư vấn.
                        </p>
                    </div>
                </div>
            </div>
            <div class="card my-3" style="width: 19rem">
                <img src="./public/img/giaiphap3.png" class="card-img-top" alt="..." />
                <div class="card-body">
                    <div class="card-text">
                        <h3 style="font-size: 22px">Đặt lịch hẹn</h3>
                        <p style="text-align: justify">
                            Bạn có thể đặt lịch hẹn với bác sĩ hoặc phòng khám. Gọi điện
                            thoại trực tiếp hoặc sử dụng hệ thống đặt lịch hẹn trực
                            tuyến
                        </p>
                    </div>
                </div>
            </div>
            <div class="card my-3" style="width: 19rem">
                <img src="./public/img/giaiphap4.png" class="card-img-top" alt="..." />
                <div class="card-body">
                    <div class="card-text">
                        <h3 style="font-size: 22px">Nhận được giải pháp</h3>
                        <p style="text-align: justify">
                            Bạn có thể nhận được chỉ đạo về thuốc, liệu pháp, phẫu thuật
                            hoặc các biện pháp khác để giải quyết vấn đề sức khỏe của
                            bạn.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="bacsi" style="
          background-image: url(https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/newest-articles-background.svg);
        ">
    <div class="container">
        <h1 class="my-5 text-center">Bác sĩ nổi bật</h1>
        <div class="separator-purple"></div>
        <div class="row">
        <?php
            if (isset($data_bs) and $data_bs != NULL) {
                foreach ($data_bs as $value) {
                    ?>
            <div class="card col-md-6">
                <img src="public/img/images/doctor/<?php echo $value['hinhanh']?>"
                    class="card-img-top" alt="<?php echo $value['hinhanh']?>" />
                <div class="card-body">
                    <h5 class="card-title"><?php echo $value['name']?></h5>
                    <p class="card-text">Giám đốc Trung tâm Tim mạch</p>
                    <div class="group-button">
                        <a href="?act=detail&id=<?=$value['id_bs']?>" class="button-blue">Xem chi tiết</a>
                        <a href="?act=detail&id=<?=$value['id_bs']?>" class="button-purple">Đặt lịch hẹn</a>
                    </div>
                </div>
            </div>
            <!-- <div class="card col-md-6">
                <img src="https://tamanhhospital.vn/wp-content/uploads/2021/03/bs-nguyen-ba-my-nhi.png"
                    class="card-img-top" alt="..." />
                <div class="card-body">
                    <h5 class="card-title">BS.CKII NGUYỄN BÁ MỸ NHI</h5>
                    <p class="card-text">Giám đốc Trung tâm Sản phụ khoa</p>
                    <div class="group-button">
                        <a href="#" class="button-blue">Xem chi tiết</a>
                        <a href="" class="button-purple">Đặt lịch hẹn</a>
                    </div>
                </div>
            </div> -->
            
            <?php }
            } else {
                echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
            }
            ?>
        </div>
    </div>
</div>
<div class="nhathuoc">
    <div class="imgage-bg">
        <img src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/images/online-pharmacy-video-cover.svg"
            alt="" />
    </div>
    <div class="imgage-shipper">
        <img src="./public/img/shipper.png" alt="" />
    </div>

    <div class="content">
        <h2 class="my-3">Nhà Thuốc Trực Tuyến</h2>
        <div class="thanhngang my-3"></div>
        <p class="my-3">
            Dễ dàng đặt trực tuyến thuốc và các sản phẩm chăm sóc sức khỏe chính
            hãng với mức giá tiết kiệm
        </p>
        <div class="row my-4">
            <div class="col">
                <img src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/icons/delivery-icon.svg"
                    alt="" />
                Giao hàng nhanh trong 2 giờ
            </div>
            <div class="col">
                <img src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/icons/discount-icon.svg"
                    alt="" />
                Tiết kiệm 20% tất cả các sản phẩm
            </div>
        </div>
        <div style="display: flex; justify-content: center" class="my-5">
            <a href="?act=pharmacy" class="text-center button-green" style="width: 150px; height: 50px">
                Xem ngay
            </a>
        </div>
    </div>
</div>
<div class="trainghiem">
    <div class="container">
        <h1 class="text-center my-5">Trải nghiệm khách hàng</h1>
        <div class="separator-purple"></div>
        <div class="trainghiem-list">
            <div class="trainghiem__col">
                <div class="trainghiem__item">
                    <div class="trainghiem__cont">
                        <h3 class="trainghiem__name">
                            Tôi đã chữa khỏi mỡ máu như thế nào?
                        </h3>
                        <p class="p1">anh Huy - GĐ công ty xây dựng</p>
                        <p class="p2">
                            Tôi là giám đốc, thường xuyên phải đi liên hoan với đối tác.
                            Các bữa ăn có rất nhiều thịt thà, đồ ăn giàu chất béo, lại
                            còn phải uống…
                        </p>
                        <div class="iconLeft">
                            <i class="fa-solid fa-quote-left"></i>
                        </div>
                        <div class="iconRight">
                            <i class="fa-solid fa-quote-right"></i>
                        </div>
                    </div>
                    <div class="trainghiem__avatar">
                        <img src="https://www.mediplus.vn/wp-content/uploads/2021/05/Khach-hang-phan-binh-1.jpg"
                            alt="" />
                    </div>
                </div>
            </div>
            <div class="trainghiem__col">
                <div class="trainghiem__item">
                    <div class="trainghiem__cont">
                        <h3 class="trainghiem__name">
                            Việc chăm con đã trở nên dễ dàng hơn
                        </h3>
                        <p class="p1">chị Huyền - nội trợ</p>
                        <p class="p2">
                            Là một bà mẹ trẻ, tôi luôn lo lắng và muốn đảm bảo sức khỏe
                            tốt nhất cho bé sơ sinh của mình. Tuy nhiên, tôi gặp khá
                            nhiều khó khăn...
                        </p>
                        <div class="iconLeft">
                            <i class="fa-solid fa-quote-left"></i>
                        </div>
                        <div class="iconRight">
                            <i class="fa-solid fa-quote-right"></i>
                        </div>
                    </div>
                    <div class="trainghiem__avatar">
                        <img src="https://www.mediplus.vn/wp-content/uploads/2021/05/khach-hang-thanh-huyen-56-tuoi-1.jpg"
                            alt="" />
                    </div>
                </div>
            </div>
            <div class="trainghiem__col">
                <div class="trainghiem__item">
                    <div class="trainghiem__cont">
                        <h3 class="trainghiem__name">
                            Căng thẳng mệt mỏi vì công việc đã giảm
                        </h3>
                        <p class="p1">anh Hùng - nhân viên văn phòng</p>
                        <p class="p2">
                            Tôi là giám đốc, thường xuyên phải đi liên hoan với đối tác.
                            Các bữa ăn có rất nhiều thịt thà, đồ ăn giàu chất béo, lại
                            còn phải uống…
                        </p>
                        <div class="iconLeft">
                            <i class="fa-solid fa-quote-left"></i>
                        </div>
                        <div class="iconRight">
                            <i class="fa-solid fa-quote-right"></i>
                        </div>
                    </div>
                    <div class="trainghiem__avatar">
                        <img src="https://www.mediplus.vn/wp-content/uploads/2021/05/khach-hang-hoang-nam-1.jpg"
                            alt="" />
                    </div>
                </div>
            </div>
            <div class="trainghiem__col">
                <div class="trainghiem__item">
                    <div class="trainghiem__cont">
                        <h3 class="trainghiem__name">
                            Tìm ra giải pháp cho vấn đề tiểu đường
                        </h3>
                        <p class="p1">anh Minh - kĩ sư phần mềm</p>
                        <p class="p2">
                            Với công việc áp lực và lối sống không lành mạnh, tôi đã bị
                            mắc bệnh tiểu đường. Tôi đã tìm đến trang web tư vấn y tế và
                            rất vui khi...
                        </p>
                        <div class="iconLeft">
                            <i class="fa-solid fa-quote-left"></i>
                        </div>
                        <div class="iconRight">
                            <i class="fa-solid fa-quote-right"></i>
                        </div>
                    </div>
                    <div class="trainghiem__avatar">
                        <img src="https://www.mediplus.vn/wp-content/uploads/2021/05/khach-hang-tran-quang-65-tuoi-1.jpg"
                            alt="" />
                    </div>
                </div>
            </div>
            <div class="trainghiem__col">
                <div class="trainghiem__item">
                    <div class="trainghiem__cont">
                        <h3 class="trainghiem__name">
                            Thông tin y tế chính xác và đáng tin cậy
                        </h3>
                        <p class="p1">anh Linh - sinh viên y khoa</p>
                        <p class="p2">
                            Là một sinh viên y khoa, việc tìm kiếm thông tin y tế chính
                            xác và đáng tin cậy là rất quan trọng đối với tôi. Trang web
                            tư vấn y tế đã giúp tôi có được những kiến...
                        </p>
                        <div class="iconLeft">
                            <i class="fa-solid fa-quote-left"></i>
                        </div>
                        <div class="iconRight">
                            <i class="fa-solid fa-quote-right"></i>
                        </div>
                    </div>
                    <div class="trainghiem__avatar">
                        <img src="https://www.mediplus.vn/wp-content/uploads/2021/05/khach-hang-hoang-nam-1.jpg"
                            alt="" />
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="hotro">
    <div class="container">
        <h1 class="text-center my-5">Các hình thức hỗ trợ</h1>
        <div class="separator-purple"></div>
        <img class="bg"
            src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/images/health-resource-background.svg"
            alt="" style="z-index: -1;"/>
        <div class="row" style="margin-top: 100px">
            <div class="col-3 col-lg-3 col-md-6 col-sm-8 mx-sm-auto">
                <div class="card">
                    <img src="./public/img/support-icon.png" class="card-img-top" alt="..." />
                    <div class="card-body text-center">
                        <h4>Hỗ trợ đặt khám</h4>
                        <p class="card-text">1900-1900</p>
                    </div>
                </div>
            </div>
            <div class="col-3 col-lg-3 col-md-6 col-sm-8 mx-sm-auto">
                <div class="card">
                    <img src="./public/img/qrface.png" class="card-img-top" alt="..." />
                    <div class="card-body text-center">
                        <h4>Facebook</h4>
                    </div>
                </div>
            </div>
            <div class="col-3 col-lg-3 col-md-6 col-sm-8 mx-sm-auto">
                <div class="card">
                    <img src="./public/img/qr-zalo.png" class="card-img-top" alt="..." />
                    <div class="card-body text-center">
                        <h4>Zalo</h4>
                    </div>
                </div>
            </div>
            <div class="col-3 col-lg-3 col-md-6 col-sm-8 mx-sm-auto">
                <div class="card">
                    <img src="./public/img/qr-mess.png" class="card-img-top" alt="..." />
                    <div class="card-body text-center">
                        <h4>Messenger</h4>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>