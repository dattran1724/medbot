<body id="page-top">

    <!-- Page Wrapper -->
    <div id="wrapper">
        <?php require_once('./views/inc/navbar.php'); ?>
        <!-- Content Wrapper -->
        <div id="content-wrapper" class="d-flex flex-column">

            <!-- Main Content -->
            <div id="content">

                <!-- Topbar -->
                <?php require_once('./views/inc/topbar.php'); ?>
                <!-- End of Topbar -->

                <!-- Begin Page Content -->
                <div class="container-fluid">
                    <div class="card shadow mb-4">
                        <div class="card-header py-3">
                            <h6 class="m-0 font-weight-bold text-primary">Edit Doctor</h6>
                        </div>
                        <?php
                        if (isset($data) and $data != NULL) {
                            foreach ($data as $value) {
                                $formattedDate = date("Y-m-d", strtotime(str_replace('/', '-', $value['ngay'])));
                                $mangGio = explode(' - ', $value['gio']);
                                if (count($mangGio) == 2) {
                                    $gioBatDau = $mangGio[0];
                                    $gioKetThuc = $mangGio[1];
                                }
                                ?>
                                <div class="card-body">
                                    <form method="POST" id="edit"
                                        action="?act=status&process=update&id=<?php echo $value['id_status']; ?>">
                                        <div class="form-group mb-3">
                                            <span class="">ID:</span>
                                            <input type="text" name="" class="form-control"
                                                value="<?php echo $value['id_status'] ?>" disabled>
                                        </div>
                                        <div class="form-group mb-3">
                                            <input type="date" name="ngay" class="form-control" placeholder="Day"
                                                value="<?php echo $formattedDate ?>">
                                        </div>
                                        <div class="form-group mb-3">
                                            <input type="time" name="giobatdau" class="form-control" placeholder="Time start"
                                                value="<?php echo $gioBatDau?>">
                                        </div>
                                        <div class="form-group mb-3">
                                            <input type="time" name="gioketthuc" class="form-control" placeholder="Time end"
                                                value="<?php echo $gioKetThuc ?>">
                                        </div>
                                    </form>
                                </div>
                            <?php }
                        } else {
                            echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
                        }
                        ?>
                        <div class=" mx-auto justify-content-center" style="width: 200px;">
                            <button class="col-2 col-sm btn btn-lg btn-primary my-1 mx-1" type="submit"
                                form="edit">Confirm</button>
                        </div>
                        <script>
                    </div >
                </div >


            </div >
            <!-- End of Main Content -->

</body >