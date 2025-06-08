
<div >
<div class="ck">
    <?php
    if (isset($data) and $data != NULL) {
        foreach ($data as $value) {
            ?>
            <div class="container position-relative p-0">
                <div>
                    <img alt="Cơ Xương Khớp" fetchpriority="high" decoding="async" data-nimg="fill"
                        style="position:absolute;height:100%;width:100%;left:0;top:0;right:0;bottom:0;object-fit:cover;color:transparent"
                        sizes="50vw"
                        src="./public/img/images/specialist/<?php echo $value['hinhanh']?>">
                </div>
                <div class="position-relative" style="background-image:linear-gradient(3.141592653589793rad,#ffffffcc,#ffffffe5,#ffffff);">
                <h1 class>
                    <?php echo $value['name'] ?>
                </h1>
                <div id="collapsedaotao" class="  font_hel sz_15 panel-collapse collapse in" role="tabpanel"
                    aria-labelledby="headingOne">
                    <div class="view_limit">
                        <div class='view'>
                            <?php echo $value['mota'] ?>
                        </div>
                        <u class="limit_more div_hover cl_33">Xem thêm <i class="fa fa-angle-double-down"
                                aria-hidden="true"></i>
                        </u>
                    </div>

                </div>
                </div>

            </div>
            <div class="position-relative" style = "background-color : #eee">
                <div class = "container">
                    <?php
                    if (isset($data_bs) and $data_bs != NULL) {
                        foreach ($data_bs as $value) {
                            ?>
                            <div class="content__main">
                                <div class="content__row">
                                    <div class="row__picture">
                                        <a href="?act=detail&id=<?= $value['id_bs'] ?>">
                                            <img src="./public/img/images/doctor/<?php echo $value['hinhanh'] ?>"
                                                alt="<?php echo $value['hinhanh'] ?>" />
                                        </a>

                                    </div>
                                    <div class="row__info">
                                        <div class="row__title">
                                            <a href="?act=detail&id=<?= $value['id_bs'] ?>" title="<?php echo $value['hinhanh'] ?>">
                                                <h2>
                                                    <?php echo $value['name'] ?>
                                                </h2>
                                            </a>
                                        </div>
                                        <div class="info__small">
                                            <div>
                                                <?php echo $value['chucvu'] ?>
                                            </div>
                                            <div>
                                                <?php echo $value['lamviec'] ?>
                                            </div>
                                        </div>
                                        <div class="info__decription">
                                            <?php echo $data1 = substr($value['mota'], 0, 350) . "..." ?>
                                        </div>
                                        <div class="info__button">
                                            <a href="?act=detail&id=<?= $value['id_bs'] ?>" class="button-purple">Đặt lịch hẹn</a>

                                            <a href="?act=detail&id=<?= $value['id_bs'] ?>" class="button-blue">Xem chi tiết
                                            </a>

                                        </div>
                                    </div>
                                </div>
                            </div>


                        <?php }
                    } else {
                        echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
                    }
                    ?>
                </div>                

            </div>
        <?php }
    } else {
        echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
    }
    ?>
</div>
</div>
