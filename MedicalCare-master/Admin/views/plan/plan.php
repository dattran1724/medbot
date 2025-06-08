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
                    <!-- DataTales Example -->
                    <div class="card shadow mb-4">
                        <div class="card-header py-3">
                            <h6 class="m-0 font-weight-bold text-primary">Plan</h6>
                        </div>

                        <div class="card-body">
                            <div class="my-2">
                                <button type="button" class="btn btn-primary mx-1 my-1" data-bs-toggle="modal"
                                    data-bs-target="#addPlanModal">
                                    Add Plan
                                </button>
                            </div>
                            <!-- Modal -->
                            <div class="modal fade" id="addPlanModal" tabindex="-1" aria-labelledby="exampleModalLabel"
                                aria-hidden="true">
                                <div class="modal-dialog modal-lg">
                                    <div class="modal-content">
                                        <div class="modal-header">
                                            <h1 class="modal-title fs-5" id="exampleModalLabel">Plan</h1>
                                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                                aria-label="Close"></button>
                                        </div>
                                        <div class="modal-body">
                                            <form id="add" class="user mx-5" action="?act=plan&process=store"
                                                method="post">
                                                <select class="form-select mb-3" aria-label="Default select example"
                                                    name="id_bs">
                                                    <?php foreach ($data_bs as $vl) { ?>
                                                        <?php if ($vl['id_bs'] == $value['id_bs']) { ?>
                                                            <option value="<?php echo $vl['id_bs'] ?>" selected>
                                                                <?php echo $vl['name'] ?>
                                                            </option>
                                                        <?php } else { ?>
                                                            <option value="<?php echo $vl['id_bs'] ?>">
                                                                <?php echo $vl['name'] ?>
                                                            </option>
                                                        <?php }
                                                    } ?>
                                                </select>
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
                                        <div class="modal-footer">
                                            <button type="button" class="btn btn-secondary"
                                                data-bs-dismiss="modal">Close</button>
                                            <button type="submit" class="btn btn-primary btn-user" form="add">Add
                                                Plan</button>
                                        </div>

                                    </div>
                                </div>


                            </div>
                            <div class="row">
                                <div class="table-responsive">
                                    <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                                        <thead>
                                            <tr>
                                                <th>ID</th>
                                                <th>Doctor</th>
                                                <th>Date</th>
                                                <th>Time</th>
                                                <th>Status</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <?php foreach ($data as $value): ?>
                                                <tr>
                                                    <td>
                                                        <?php echo $value['id_kh'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['name'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['ngay'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['gio'] ?>
                                                    </td>
                                                    <td>
                                                        <?php if ($value['trangthai'] == 0) { ?>
                                                            <div class="alert alert-info">Pending</div>
                                                        <?php } else { ?>
                                                            <div class="alert alert-success">Received</div>
                                                        <?php } ?>
                                                    </td>
                                                    <td>
                                                        <div class="row mx-1 my-0">
                                                            <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1"
                                                                href="?act=plan&process=edit&id=<?php echo $value['id_kh']; ?>">Edit</a>
                                                            <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1"
                                                                href="?act=plan&process=delete&id=<?php echo $value['id_kh']; ?>">Delete</a>
                                                        </div>
                                                    </td>
                                                </tr>
                                            <?php endforeach; ?>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <!-- /.container-fluid -->

            </div>
            <!-- End of Main Content -->

</body>