<!-- <div class="div_breadcrumb xxsmall_mb  pt_10 pb_10 text-uppercase small"style="position: relative;z-index: 1;">
    <div class="container">
        <div class="row">
            <div class="col-sm-12">
                <nav aria-label="breadcrumbs" class="rank-math-breadcrumb">
                    <p><a href="?act=home">Trang chủ</a><span class="separator"> &gt; </span><span
                            class="last">Chuyên gia - bác sĩ</span></p>
                </nav>
            </div>
        </div>
    </div>
</div> -->
<div class="care-service-top" style="margin-top: 80px">

    <div class="care-service-background"></div>
    <div class="care-service-banner">
        <div class="top-care-service-content">
            <div class="inner">
                <div class="doctor-list-title">
                    <h1 class="title">
                        Đội ngũ Bác sĩ ưu tú
                        <span class="text-nowrap">từ các</span> Bệnh viện
                        <span class="text-nowrap"> hàng đầu</span>
                    </h1>
                    <div class="covid-vaccinated media">
                        <div class="media-left">
                            <img alt="vaccine" class="covid-vac" src="./public/img/images/icons/covid-vaccine.svg" />
                            <img alt="vaccine" class="tick" src="./public/img/images/icons/vaccine-green-tick.svg" />
                        </div>
                        <div class="media-body">
                            <p>
                                Y bác sĩ của chúng tôi đã được tiêm vaccine phòng ngừa
                                COVID-19
                            </p>
                        </div>
                    </div>
                    <p class="doctor-list-info">
                        Đội ngũ Bác sĩ ưu tú với thâm niên trung bình 10 năm kinh nghiệm
                        hiện công tác tại các Bệnh viện hàng đầu Việt Nam, thăm khám
                        trên nhiều chuyên khoa đa dạng, tận tâm chăm sóc bạn cùng gia
                        đình.
                    </p>
                </div>
            </div>
        </div>
        <div class="banner-image">
            <div class="d-xl-block">
                <img class="layer-1" src="./public/img/background/doctor-list-banner-dt.svg" />
                <img class="layer-2" src="./public/img/background/doctor-list-banner-2-dt.png" />
                <img class="layer-3" src="./public/img/background/doctor-list-banner-3-dt.svg" />
            </div>
        </div>
    </div>
</div>

<div class="filter-profesor">
    <div class="container">
        <div class="search text-center">
            <div class="input-group mb-4 my-5">
                <span class="input-group-text icon-search"><i class="fa-solid fa-magnifying-glass"></i></span>
                <input type="text" class="form-control" name="search" placeholder="" />
                <span class="input-group-text search"><a href="">Search</a></span>
            </div>
        </div>
        <h1 class="text-center my-5">Chuyên gia - bác sĩ</h1>
        <div class="separator-purple"></div>
        <!-- <div class="filter__location my-3">
            <label>Toàn hệ thống</label>
            <label>Hà Nội</label>
            <label>TP.HCM</label>
        </div> -->
        <!-- <select class="filter__location my-3" id="locationFilter">
        <option value="all">Toàn hệ thống</option>
        <option value="hanoi">Hà Nội</option>
        <option value="hochiminh">TP.HCM</option>
        </select> -->
        <!-- <div class="filter__fill my-3">
            <select name="" id="" class="speciallist">
                <option>Chuyên khoa</option>
                <option value="">Khoa ngoại vú</option>
                <option value="">Khoa tiết niệu</option>
                <option value="">Khoa hô hấp</option>
                <option value="">Khoa thần kinh</option>
                <option value="">Khoa cơ xương khớp</option>
                <option value="">Khoa da liễu</option>
            </select>
            <select name="" id="" class="regency">
                <option>Chức vụ</option>
                <option value="">Giám đốc</option>
                <option value="">Phó giám đốc</option>
                <option value="">Trưởng khoa</option>
                <option value="">Phó khoa</option>
                <option value="">Bác sĩ</option>
            </select>
            <select name="" id="" class="academic-rank">
                <option>Học hàm</option>
                <option value="">Giáo sư</option>
                <option value="">Phó giáo sư</option>
            </select>
            <select name="" id="" class="degree">
                <option>Học vị</option>
                <option value="">Thạc sĩ</option>
                <option value="">Tiến sĩ</option>
                <option value="">Bác sĩ nội trú</option>
                <option value="">Bác sĩ</option>
                <option value="">Bác sĩ CKI</option>
                <option value="">Bác sĩ cao cấp</option>
                <option value="">Dược sĩ</option>
            </select>
        </div> -->
    </div>
</div>
<div class="content">
    <div class="container">
        <?php require('list-bacsi.php') ?>
    </div>
</div>
<div class="row mt-5">
    <div class="col-sm-12">
        <div class="pagnation-ul mb-5">
            <ul class="clearfix">
                <?php if ($data_tong > 0) {
                    for ($i = 1; $i <= ceil($data_tong / 6); $i++) { ?>
                        <li><a href="?act=chuyengia&trang=<?= $i ?>">
                                <?= $i ?>
                            </a></li>
                    <?php }
                }
                ?>

            </ul>
        </div>
    </div>
</div>