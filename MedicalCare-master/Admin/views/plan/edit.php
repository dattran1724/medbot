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
                                ?>
                                <div class="card-body">
                                    <form method="POST" id="edit"
                                        action="?act=plan&process=update&id=<?php echo $value['id_kh']; ?>">
                                        <div class="form-group mb-3">
                                            <span class="">ID:</span>
                                            <input type="text" name="" class="form-control"
                                                value="<?php echo $value['id_kh'] ?>" disabled>
                                        </div>
                                        <div class="form-group mb-3">
                                            <input type="text" name="id_bs" class="form-control" placeholder="Day"
                                                value="<?php echo $value['name']?>" disabled>
                                        </div>
                                        <select class="form-select mb-3" aria-label="Default select example"
                                                    name="id_status">
                                                    <?php foreach ($data_status as $ck) { ?>
                                                        <?php if ($ck['id_status'] == $value['id_status']) { ?>
                                                            <option value="<?php echo $ck['id_status'] ?>" selected>
                                                                <?php echo $ck['ngay'] .' - '. $ck['gio'] ?>
                                                            </option>
                                                        <?php } else { ?>
                                                            <option value="<?php echo $ck['id_status'] ?>">
                                                                <?php echo $ck['ngay'] .' - '. $ck['gio'] ?>
                                                            </option>
                                                        <?php }
                                                    } ?>
                                                </select>
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