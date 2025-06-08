<?php
    if (isset($data) and $data != NULL) {
        foreach ($data as $value) {
            ?>
<!-- <div class="div_breadcrumb xxsmall_mb  pt_10 pb_10 text-uppercase small"style="position: relative;z-index: 1;">
    <div class="container">
        <div class="row">
            <div class="col-sm-12">
                <nav aria-label="breadcrumbs" class="rank-math-breadcrumb">
                    <p><a href="?act=home">Trang chủ</a><span class="separator"> &gt; </span>
                    <a href="?act=chuyengia">Chuyên gia - bác sĩ</a>
                    <span class="separator"> &gt; </span>
                    <span class="last"><?php echo $value['name'] ?></span>    
                </p>
                </nav>
            </div>
        </div>
    </div>
</div> -->
<div class="container">
            <div class="bs-thongtin ">
                <div class="vung-bao">
                    <div class="bs-anh">
                        <a href="?act=detail&id=<?= $value['id_bs'] ?>">
                            <img src="public/img/images/doctor/<?php echo $value['hinhanh'] ?>">
                        </a>
                    </div>
                    <div class="bs-soluoc">
                        <h1 class="bs-ten my-3">
                            <?php echo $value['name'] ?>
                        </h1>
                        <div class="bs-tomtat">
                            <?php echo $value['chucvu'] ?><br>
                            <?php echo $value['lamviec'] ?>
                        </div>
                    </div>
                </div>
            </div>
            <div class="bs-lichkham bs-phandoan"> 
                <div id="lichkham">
                    <div class="vung-bao">
                        <div id="lichkham_chitiet" class="lichkham_chitiet uutien-som">
                            <div class="mot-bs ">
                                <div class="mot-bs-lichkham">
                                    <p class="mot-bs-chon-gio">
                                        <select id="dateSelector" class="select" onchange="toggleAppointmentInfo()">
                                            <?php foreach ($ngay as $val) {
                                                echo '<option value="' . $val['ngay'] . '">';
                                                echo $val['ngay'];
                                                echo "</option> ";
                                            } ?>
                                        </select>
                                    </p>
                                    <?php
                                    $chuyengia_model = new Detail();
                                    foreach ($ngay as $val) {
                                        $gio = $chuyengia_model->giotheongay($_GET['id'], $val['ngay']);
                                        ?>
                                        <div class="mot-bs-ngay mot-bs-thu1 lich-c1466-d4 lich-c1466-d4-p154 hien"
                                            data-date="<?php echo $val['ngay'] ?>">
                                            <h3 class="mot-bacsi-dichvu mot-bacsi-dichvu-1">
                                                <i class="fa-solid fa-calendar"></i>
                                                <i class="icon-dichvu bt-g bt-g-lich "></i> Lịch Khám
                                            </h3>
                                            <div class="khung-lich mt-3">
                                                <div class="lich-kham-hom-nay">
                                                    <div class="mot-bs-co-lichkham">
                                                        <?php foreach ($gio as $g) { ?>
                                                            <a href="?act=lichhen&KH=<?php echo $g['id_kh'] ?>"
                                                                class="mot-bs-giokham mot-bs-giokham-dichvu-1 my-2">
                                                                <?php echo $g['gio'] ?>
                                                            </a>
                                                        <?php } ?>
                                                        <div class="mot-bs-lichkham-huongdan mt-3" data-name="">
                                                            Chọn và đặt(Phí đặt lịch 0<sup>đ</sup>)
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="lichkham-thongtin">
                                                    <div class="lichkham-diadiem">
                                                        <h3>Địa chỉ Khám</h3>
                                                        <div class="lichkham-ten">Phòng khám Đa khoa </div>
                                                        <div class="lichkham-diachi">470 - Trần Đại Nghĩa - Ngũ Hành Sơn - Đà Nẵng</div>
                                                    </div>
                                                    <div class="lichkham-giakham khuyen-mai">
                                                        <h3 id="tieu_de_gia" class="m-0">Giá Khám:
                                                        </h3>
                                                        <span class="lichkham-giakham-tomtat">&nbsp;300.000<sup>đ</sup>
                                                        </span>
                                                        <p class="gach m-0"> - </p>
                                                        <span class="lichkham-giakham-tomtat">400.000<sup>đ</sup></span>
                                                        <!-- <span id="cham">.</span> <a class="lichkham-banggia-hien" href="">&nbsp;Xem
                                                            chi
                                                            tiết</a>
                                                        <div class="width-100">
                                                            <div class="lichkham-banggia-bang lichkham-bang">

                                                            </div>
                                                            <a class="lichkham-banggia-an" href="#angiakham">Ẩn bảng giá</a>
                                                        </div> -->
                                                    </div>

                                                    <div class="lichkham-baohiem">
                                                        <h3>Loại bảo hiểm áp dụng.</h3>
                                                        <a class="" id="toggleLink" data-toggle="collapse" href="#collapseContent"
                                                            role="button" aria-expanded="false" aria-controls="collapseExample">
                                                            Xem chi tiết
                                                        </a>

                                                        <div id="collapseContent" class="collapse">
                                                            <div class="baohiem-content">
                                                                <!-- Nội dung của bạn vào đây -->
                                                                <div class="insurance-info">
                                                                    <h4 style="font-size: 14px">Bảo hiểm y tế nhà nước</h4>
                                                                    <div class="additional-info">Hiện chưa áp dụng bảo hiểm y tế nhà
                                                                        nước cho dịch vụ khám chuyên gia.</div>
                                                                </div>
                                                                <div class="insurance-info">
                                                                    <h4 style="font-size: 14px">Bảo hiểm y tế tư nhân </h4>
                                                                    <div class="additional-info">Đối với các đơn vị bảo hiểm
                                                                        không bảo lãnh trực tiếp, phòng khám xuất hoá đơn tài
                                                                        chính (hoá đơn đỏ) và hỗ trợ bệnh nhân hoàn thiện hồ
                                                                        sơ</div>
                                                                </div>
                                                            </div>
                                                            <a class="" id="toggleLinkHide" data-toggle="collapse"
                                                                href="#collapseContent" role="button" aria-expanded="false"
                                                                aria-controls="collapseExample">
                                                                Ẩn
                                                            </a>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            <div class="xeplai"></div>
                                        </div>
                                    <?php } ?>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <?php require('content.php') ?>
        <?php }
    } else {
        echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
    }
    ?>
</div>