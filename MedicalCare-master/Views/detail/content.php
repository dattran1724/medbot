<div class="bg_head z99">
            <div class="container div_section">
                <div class="row">
                    <div class="col-xs-12 noleft_mb noright_mb ">
                        <div
                            class="cl_white ul_not div_flex div_flex_mobile text-uppercase pt_15 pb_15 ml_t15 mr_t15 div_board">
                            <h2 class="sz_15 mt_0 mb_0 font_hel div_inline"><span
                                    class="pl_15 pr_15 pl_mb_5 pr_mb_5 click_cgia item_hover div_box active"
                                    id="cgia_gioithieu" data-cgia="cgia_gioithieu"><a href="#cgia_gioithieu"><span
                                            class="sz_15">Giới thiệu</span></a></span></h2>
                            <!-- <h2 class="sz_15 mt_0 mb_0 font_hel div_inline"><span
                                    class="pl_15 pr_15 pl_mb_5 pr_mb_5 click_cgia item_hover div_box" id="cgia_thanhtuu"
                                    data-cgia="cgia_thanhtuu"><a href="#cgia_thanhtuu"><span class="sz_15">Thành
                                            tựu</span></a></span></h2> -->
                            <h2 class="sz_15 mt_0 mb_0 font_hel div_inline"><span
                                    class="pl_15 pr_15 pl_mb_5 pr_mb_5 click_cgia item_hover div_box"
                                    id="cgia_chuyenmon" data-cgia="cgia_chuyenmon"><a href="#cgia_chuyenmon"><span
                                            class="sz_15">Chuyên môn</span></a></span></h2>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="content_chuyengia pt_15 ">
            <div class="container">
                <div class="row">
                    <div class="col-xs-12 pb_15">
                        <div class="click_item cgia_gioithieu ">
                            <h3 class="cl_head mt_0 font_hel sz_15 text-uppercase line_head mb_10 pb_5" role="button"
                                data-toggle="collapse" data-parent="#accordion" href="#collapsegioithieu"
                                aria-expanded="true" aria-controls="collapsegioithieu"><img class='icon_left lazyload'
                                    data-src='https://tamanhhospital.vn/icon/i_user.jpg' alt=''>GIỚI THIỆU
                            </h3>
                            <div id="collapsegioithieu" class="panel-collapse collapse in mb_10" role="tabpanel"
                                aria-labelledby="headingOne">
                                <div class='view_limit'>
                                        <?php echo $value['mota'] ?>
                                    <u class="limit_more div_hover cl_33">xem thêm <i class="fa fa-angle-double-down"
                                            aria-hidden="true"></i></u>
                                </div>
                            </div>
                            <?php if ($value['thanhvien'] != "NULL") { ?>
                            <h3 class="cl_head mt_0 font_hel sz_15 text-uppercase line_head mb_10 pb_5" role="button"
                                data-toggle="collapse" data-parent="#accordion" href="#collapsethanhvien"
                                aria-expanded="true" aria-controls="collapsethanhvien"><img class='icon_left lazyload'
                                    data-src='https://tamanhhospital.vn/icon/i_group.jpg' alt=''>THÀNH VIÊN TỔ CHỨC
                            </h3>
                            
                            <div id="collapsethanhvien" class="panel-collapse collapse in mb_10" role="tabpanel"
                                aria-labelledby="headingOne">
                                <div class='view_limit'>
                                        <?php echo $value['thanhvien'] ?>
                                    <u class="limit_more div_hover cl_33">Xem thêm <i class="fa fa-angle-double-down"
                                            aria-hidden="true"></i></u>
                                </div>
                            </div>
                            <?php } ?>
                        </div>
                        <!-- <div class="click_item cgia_thanhtuu hidden">
                            <h3 class="cl_head mt_0 font_hel sz_15 text-uppercase line_head mb_10 pb_5" role="button"
                                data-toggle="collapse" data-parent="#accordion" href="#collapsedanhhieu"
                                aria-expanded="true" aria-controls="collapsedanhhieu"><img class='icon_left lazyload'
                                    data-src='https://tamanhhospital.vn/icon/i_danhhieu.jpg' alt=''>DANH HIỆU, GIẢI
                                THƯỞNG<i class="fa fa-minus-circle cl_head bg_white pull-right mt_5"
                                    aria-hidden="true"></i></h3>
                            <div id="collapsedanhhieu" class="panel-collapse collapse in mb_10" role="tabpanel"
                                aria-labelledby="headingOne">
                                <div class='view_limit'>
                                    
                                    <u class="limit_more div_hover cl_33">xem thêm <i class="fa fa-angle-double-down"
                                            aria-hidden="true"></i></u>
                                </div>
                            </div>
                            <h3 class="cl_head mt_0 font_hel sz_15 text-uppercase line_head mb_10 pb_5" role="button"
                                data-toggle="collapse" data-parent="#accordion" href="#collapsenghiencuu"
                                aria-expanded="true" aria-controls="collapsenghiencuu"><img class='icon_left lazyload'
                                    data-src='https://tamanhhospital.vn/icon/i_nghiencuu.jpg' alt=''>CÔNG TRÌNH NGHIÊN
                                CỨU<i class="fa fa-minus-circle cl_head bg_white pull-right mt_5"
                                    aria-hidden="true"></i></h3>
                            <div id="collapsenghiencuu" class="panel-collapse collapse in mb_10" role="tabpanel"
                                aria-labelledby="headingOne">
                                <div class='view_limit'>
                                    
                                    <u class="limit_more div_hover cl_33">xem thêm <i class="fa fa-angle-double-down"
                                            aria-hidden="true"></i></u>
                                </div>
                            </div>
                            <div id="collapsecathanhcong" class="panel-collapse collapse in mb_10 pt_10" role="tabpanel"
                                aria-labelledby="headingOne">
                                <div class="row list_cathanhcong">
                                </div>
                            </div>
                        </div> -->
                        <div class="click_item cgia_chuyenmon hidden">
                            <h3 class="font_hel sz_15 cl_head text-uppercase line_head mb_10 pb_5" role="button"
                                data-toggle="collapse" data-parent="#accordion" href="#collapsedaotao"
                                aria-expanded="true" aria-controls="collapsedaotao"><img class='icon_left lazyload'
                                    data-src='https://tamanhhospital.vn/icon/i_daotao.jpg' alt=''>QUÁ TRÌNH ĐÀO TẠO
                            </h3>
                            <div id="collapsedaotao" class="  font_hel sz_15 panel-collapse collapse in mb_10"
                                role="tabpanel" aria-labelledby="headingOne">
                                <div class='view_limit'>
                                    <?php echo $value['daotao'] ?>
                                    <u class="limit_more div_hover cl_33">xem thêm <i class="fa fa-angle-double-down"
                                            aria-hidden="true"></i></u>
                                </div>
                            </div>
                            <h3 class="cl_head mt_0 font_hel sz_15 text-uppercase line_head mb_10 pb_5" role="button"
                                data-toggle="collapse" data-parent="#accordion" href="#collapsekinhnghiemct"
                                aria-expanded="true" aria-controls="collapsekinhnghiemct"><img
                                    class='icon_left lazyload'
                                    data-src='https://tamanhhospital.vn/icon/i_kinhnghiem.jpg' alt=''>KINH NGHIỆM CÔNG
                                TÁC<i class="fa fa-minus-circle cl_head bg_white pull-right mt_5"
                                    aria-hidden="true"></i></h3>
                            <div id="collapsekinhnghiemct" class=" font_hel sz_15 panel-collapse collapse in mb_10"
                                role="tabpanel" aria-labelledby="headingOne">
                                <div class='view_limit'>
                                <?php echo $value['kinhnghiem'] ?>
                                    <u class="limit_more div_hover cl_33">xem thêm <i class="fa fa-angle-double-down"
                                            aria-hidden="true"></i></u>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>