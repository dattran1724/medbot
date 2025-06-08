<?php
if (isset($data) and $data != NULL) {
    foreach ($data as $value) {
        ?>
        <div class=" mot-bs">
            <div class="container vung-bao">
                <div class="mot-bs-thongtin">
                    <div class="mot-bs-anh"><a href=""><img src="./public/img/images/doctor/<?php echo $value['hinhanh'] ?>"
                                width="200" height="200" alt="<?php echo $value['name'] ?>"></a></div>
                    <div class="mot-bs-soluoc">
                        <h1>Đặt lịch khám</h1>
                        <h2 class="mot-bs-ten"><a href="?act=detail&id=<?= $value['id_bs'] ?>">
                                <?php echo $value['name'] ?>
                            </a></h2>
                        <div class="mot-bs-diadiem">
                            <?php echo $value['ngay'];
                            echo '-';
                            echo $value['gio'] ?>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="datlich-thongtin mb-5" style="position: relative">
            <div class="container vung-bao">
                <form method="POST" id="datlichkham" action="?act=lichhen&KH=<?php echo $value['id_kh'] ?>&xuli=datlich">
                    <div>
                        <label class="giakham chon">
                            <input type=radio checked=checked /><span>Giá khám </span>
                            <div>
                                <?php echo $value['giakham'] ?>
                            </div>
                        </label>
                    </div>
                    <!-- <div class="dauvao-nhom">
                        <label>
                            <input class="dauvao-nhap" type="radio" name="book_for" value="0"> Đặt cho mình
                        </label>
                        <label>
                            <input class="dauvao-nhap" type="radio" name="book_for" value="1"> Đặt cho người thân
                        </label>
                    </div> -->
                    <div class="datcho-nguoi">
                        <!-- <p class="tieude-nhom">Thông tin người đặt lịch</p>
                        <div class="dauvao-nhom">
                            <div class="o-nhap">
                                <span class="dauvao-bt"></span>
                                <input class="dauvao-nhap" id="contact_name" name="contact_name" disabled=disabled type=text
                                    placeholder="Họ tên người đặt lịch" />
                            </div>
                        </div>
                        <div class="dauvao-nhom">
                            <div class="o-nhap"><span class="dauvao-bt"></span>
                                <input class="dauvao-nhap batbuoc" name="phone" value=""
                                    placeholder="Số điện thoại liên hệ (bắt buộc)" />
                            </div>
                            <div class="dauvao-thongbao"></div>
                        </div> -->
                        <p class="tieude-nhom">Thông tin bệnh nhân</p>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap">
                            <span class="dauvao-bt"></span>
                            <input class="dauvao-nhap batbuoc" type=text id="name" name="name"
                                placeholder="Họ tên bệnh nhân (bắt buộc)" />
                        </div>
                        <div class="dauvao-thongbao"></div>
                        <div class="dauvao-ghichu">Hãy ghi rõ Họ Và Tên, viết hoa những chữ cái đầu tiên, ví dụ: Trần Văn Phú
                        </div>
                    </div>
                    <div class="dauvao-nhom">
                        <label>
                            <input class="dauvao-nhap batbuoc" type="radio" name="gender" value="0" /> Nam
                        </label>
                        <label>
                            <input class="dauvao-nhap" type="radio" name="gender" value="1" /> Nữ
                        </label>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom datcho-nguoi">
                        <div class="o-nhap"><span class="dauvao-bt"></span>
                            <input class="dauvao-nhap" name="phone2" placeholder="Số điện thoại bệnh nhân" autocomplete=tel />
                        </div>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap"><span class="dauvao-bt-lich"></span>
                            <input class="dauvao-nhap batbuoc" name="ngaysinh" type="date" placeholder="Ngày sinh (bắt buộc)" />
                        </div>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap"><span class="dauvao-bt-diadiem"></span>
                            <select class="dauvao-nhap batbuoc" id="city" name="tinh" aria-label=".form-select-sm">
                                <option value="" selected>Chọn tỉnh thành</option>
                            </select>
                        </div>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap">
                            <span class="dauvao-bt-diadiem">

                            </span>
                            <select class="dauvao-nhap" id="district" name="huyen" aria-label=".form-select-sm">
                                <option value="" selected>Chọn quận huyện</option>
                            </select>
                        </div>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap">
                            <span class="dauvao-bt-diadiem">

                            </span>
                            <select class="dauvao-nhap" id="ward" name="xa" aria-label=".form-select-sm">
                                <option value="" selected>Chọn phường xã</option>
                            </select>
                            <div class="dauvao-thongbao"></div>
                        </div>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap">
                            <span class="dauvao-bt-diadiem"></span>
                            <input class="dauvao-nhap" name="diachi" type=text placeholder="Địa chỉ" />
                        </div>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom">
                        <div class="o-nhap">
                            <span class="dauvao-bt-cong-tron">
                            </span>
                            <textarea name="lydo" class="dauvao-nhap" placeholder="Lý do khám"></textarea>
                        </div>
                        <div class="dauvao-thongbao"></div>
                    </div>
                    <div class="dauvao-nhom thanh-toan">
                        <p class="tieude-nhom">Hình thức thanh toán</p>
                        <label>
                            <input class="dauvao-nhap" type=radio name=pay_type /> Thanh toán sau tại cơ sở y tế
                        </label>
                    </div>
                    <div class="thanhtoan-noidung" id=thanhtoan>
                        <div style="display: flex; justify-content: space-between;">
                            <div>Giá khám</div>
                            <div id=thanhtoan-giakham><?php echo $value['giakham'] ?></div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div>Phí đặt lịch</div>
                            <div id=thanhtoan-phidatlich></div>
                        </div>
                        <hr />
                        <div style="display: flex; justify-content: space-between;">
                            <div>Tổng cộng</div>
                            <div id=thanhtoan-tongcong>Chưa xác định</div>
                        </div>
                    </div>
                    <p class="dauvao-ghichu chu-giua">Quý khách vui lòng điền đầy đủ thông tin để tiết kiệm thời gian làm thủ
                        tục khám</p>
                    <div class="dauvao-canhbao">
                        <p style="text-align: justify; "><b>LƯU Ý</b></p>
                        <p style="text-align: justify; ">Thông tin anh/chị cung cấp sẽ được sử dụng làm hồ sơ khám bệnh, khi
                            điền thông tin anh/chị vui lòng:</p>
                        <ul>
                            <li style="text-align: justify; ">Ghi rõ họ và tên, viết hoa những chữ cái đầu tiên, ví dụ:
                                <b> Trần Văn Phú</b>
                            <li style="text-align: justify; ">Điền đầy đủ, đúng và vui lòng kiểm tra lại thông tin trước khi ấn
                                "Xác nhận"
                        </ul>
                    </div>
                    <?php if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) { ?>
                        <button class="button-yellow button-xacnhan" type="submit" form="datlichkham">Xác nhận đặt khám
                        </button>
                    <?php } else { ?>
                        
                        <a href="./?act=account&process=login" class="button-green button-xacnhan" style="display :block">
                             Đăng nhập trước khi đặt khám
                            
                        </a>
                        


                    <?php } ?>


                    <p class="dauvao-ghichu">Bằng việc xác nhận đặt khám, bạn đã hoàn toàn đồng ý với
                        <a href=>Điều khoản sử dụng
                        </a> dịch vụ của chúng tôi.
                    </p>
                </form>
            </div>
        </div>
        <?php
    }
} else {

    if (isset($_COOKIE['msg'])) { ?>
        <div class="container">
            <div class="thanhcong m-5"><span class="bt-g bt-g-ngoncai-len"></span>
                <i class="fa-regular fa-thumbs-up" style="font-size:50px"></i>
                <h1>
                    <?= $_COOKIE['msg'] ?>
                </h1><a href="?act=xemlichhen" class="bt-lichhen"><span>Xem lịch đã đặt</span></a>
            </div>
            <!-- <div class="alert alert-warning">
                <strong>Alert : </strong>
                <?= $_COOKIE['msg'] ?>
            </div> -->
        </div>

    <?php }

}
?>