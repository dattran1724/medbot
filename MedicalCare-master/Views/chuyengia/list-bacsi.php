<div class="tab-pane">
    <div class="tab-info">
        <?php
            if( isset($data) and $data != NULL){
                foreach($data as $value){
        ?>
        <div class="content__main">
            <div class="content__row">
                <div class="row__picture">
                    <a href="?act=detail&id=<?=$value['id_bs']?>">
                        <img src="public/img/images/doctor/<?php echo $value['hinhanh']?>" alt="<?php echo $value['hinhanh']?>" />                        
                    </a>

                </div>
                <div class="row__info">
                    <div class="row__title">
                        <a href="?act=detail&id=<?=$value['id_bs']?>" title="<?php echo $value['hinhanh']?>">
                            <h2>
                            <?php echo $value['name']?>
                            </h2>
                        </a>
                    </div>
                    <div class="info__small">
                        <div><?php echo $value['chucvu']?></div>
                        <div><?php echo $value['lamviec']?></div>
                    </div>
                    <div class="info__decription">
                        <?php echo $data1 = substr($value['mota'], 0, 350) . "..."?>
                    </div>
                    <div class="info__button">
                        <a href="?act=detail&id=<?=$value['id_bs']?>" class="button-purple">Đặt lịch hẹn</a>
                        
                        <a href="?act=detail&id=<?=$value['id_bs']?>" class="button-blue">Xem chi tiết                        
                        </a>
                        
                    </div>
                </div>
            </div>
        </div>
        <?php }}else{
			echo '<p> KHÔNG CÓ DỮ LIỆU </p>';}
        ?>
    </div>
</div>