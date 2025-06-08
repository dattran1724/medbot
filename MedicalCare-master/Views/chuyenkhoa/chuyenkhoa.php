<div class="link">
    <div class="container">
        <div class="row">
            <div class="col-sm-12">
                <!-- <p><a href="index.html">Trang chủ</a><span class=""> &gt; </span><span class="">Danh sách chuyên khoa</span></p>                   -->
            </div>
        </div>
    </div>
</div>
<div class="bn-chuyenkhoa">
    <div class="care-service-top" style="margin-top: 80px">
        <div class="care-service-banner">
            <div class="top-care-service-content">
                <div class="inner">
                    <div class="doctor-list-title">
                        <h1 class="title">
                            Đội ngũ Bác sĩ ưu tú
                            <span class="text-nowrap">từ các</span> Bệnh viện
                            <span class="text-nowrap"> hàng đầu</span>
                        </h1>
                    </div>
                    <p class="doctor-list-info">
                        Đội ngũ Bác sĩ ưu tú với thâm niên trung bình 10 năm kinh nghiệm
                        hiện công tác tại các Bệnh viện hàng đầu Việt Nam, thăm khám
                        trên nhiều chuyên khoa đa dạng, tận tâm chăm sóc bạn cùng gia
                        đình.
                    </p>
                </div>
            </div>

            <div class="banner-image">
                <div class="d-xl-block">
                    <img class="layer_1"
                        src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/banner.svg" />
                    <img class="layer_2" src="./public/img/chung-toi-theo-duoi.png" />
                    <img class="layer_3"
                        src="https://cdn.jiohealth.com/jio-website/home-page/jio-website-v2.2/assets/images/doctor-profile/doctor-list-banner-3-dt.svg" />
                </div>
            </div>
        </div>
    </div>
</div>
<div class="head text-center">
    <div class="ball lagre"></div>
    <h1 class="my-5">Danh sách chuyên khoa</h1>
    <div class="separator-purple"></div>
</div>
<div class="chuyen-khoa">
    <div class="container">
        <div class="nav-links">
            <div class="tab-content tab-content--slider">
                <div class="tab-pane">
                    <div class="tab-info">
                        <div class="cac-khoa">
                            <?php
                            if (isset($data) and $data != NULL) {
                                foreach ($data as $value) {
                                    ?>
                                    <div class="khoa-item bg-white text-center">
                                        <div class="item-box">
                                            <div class="info-khoa my-3">
                                                <a href="?act=chitietchuyenkhoa&id=<?=$value['id_ck']?>">
                                                    <div class="icon-khoa div-icon">
                                                        <img class="img-khoa"
                                                            src="./public/img/images/specialist/<?php echo $value['hinhanh']?>" alt="<?php echo $value['hinhanh']?> " />
                                                    </div>
                                                </a>
                                                <a href="?act=chitietchuyenkhoa&id=<?=$value['id_ck']?>">
                                                    <h2 class=""><?php echo $value['name']?></h2>
                                                </a>
                                            </div>
                                        </div>
                                    </div>
                                    
                                
                            <?php }
                            } else {
                                echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
                            }
                            ?></div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-sm-12">
                    <div class="pagnation-ul">
                        <ul class="clearfix">
                            <?php if ($data_tong > 0) {
                                for ($i = 1; $i <= ceil($data_tong / 12); $i++) { ?>
                                    <li><a href="?act=chuyenkhoa&trang=<?= $i ?>">
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
    </div>
</div>