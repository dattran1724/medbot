<div class="container lichhen">
    <div class=""
        style="color: rgb(16, 15, 15); font-weight: 600; font-size: 24px; margin-bottom: 15px; margin-top: 20px;">
        Lịch hẹn đã đặt</div>
    <?php
    if (isset($data) and $data != NULL) {
        foreach ($data as $value) {
            ?>
            <div class="box"
                style="z-index: 1; margin: 5px 15px 20px; border-radius: 8px; background-color: rgb(255, 255, 255); padding: 20px 30px; flex-direction: column; display: flex; box-shadow: rgba(32, 33, 36, 0.28) 0px 1px 6px;">
                <div class="" style="display: flex">
                    <div class=" " style="align-items: center; flex: 1 1 0%; margin-right: 10px;">
                        <div class=" "
                            style=" display: flex;background-color: rgb(255, 255, 255);border-radius: 50px;justify-content: center;align-items: center;margin-bottom: 5px;border-width: 1px;border-color: rgba(0, 0, 0, 0.2);flex-direction: column;">
                            <div class=" " style="width: 70px; height: 70px;">
                                <img src="./public/img/images/icons/ic_kham.b8b58dd8.png"
                                    style=" height: 100%; width: 100%; inset: 0px; object-fit: contain; color: transparent;">
                            </div>
                            <span class=""
                            style="color: rgb(69, 190, 229); text-transform: uppercase; font-size: 23px; font-weight: bold; margin-bottom: 7px; text-align: center;">Khám</span>
                        </div>
                        
                        <div class=""style="font-size: 18px;display: flex;flex-direction: column;align-items: center;">
                            <div class=" items-center justify-center" style="display: flex">
                                <div class=" h-3.5 w-3.5 xl:h-6 xl:w-6">
                                    <i class="fa-regular fa-clock"></i>
                                </div>
                                <span class=""
                                    style="color: rgb(255, 194, 14); margin-left: 5px;"><?php echo $value['gio'] ?></span>
                            </div>
                            <div class="  items-center justify-center" style="display: flex">
                                <div class="  h-3.5 w-3.5 xl:h-6 xl:w-6">
                                    <i class="fa-solid fa-calendar-days"></i>
                                </div><span class=""
                                    style="color: rgb(255, 194, 14); margin-left: 5px;"><?php echo $value['ngay'] ?></span>
                            </div>
                        </div>
                    </div>
                    <div class="detail " style="flex: 3 1 0%;font-size: 19px">
                        <div class="  mb-3 " style="">
                            <span class=" text">Bệnh nhân:
                                <span class=" text"><?php echo $value['name_bn'] ?></span>
                            </span>
                        </div>
                        <div class="  mb-3 ">
                            <span class=" text">Bác sĩ:
                                <span class="text"><?php echo $value['name_bs'] ?></span>
                            </span>
                        </div>
                        <div class="  mb-3 ">
                            <span class=" ">Nơi khám:
                                <span class=" ">Phòng khám Đa khoa </span>
                            </span>
                        </div>
                        <div class="  mb-3 ">
                            <span class="">Lý do khám:
                                <span class=" "><?php echo $value['lydo'] ?></span>
                            </span>
                        </div>
                        <div class=""
                            style=" text-align: center;width: 270px;height:45px;padding: 7px;margin-top: 10px;border-radius: 10px;align-self: flex-start;background-color: rgb(255, 194, 14);">
                            <span class="">Đã đặt khám</span>
                        </div>
                    </div>
                </div>

            </div>
        <?php }
    } ?>
</div>