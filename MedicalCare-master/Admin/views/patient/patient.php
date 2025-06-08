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
                            <h6 class="m-0 font-weight-bold text-primary">Patient</h6>
                        </div>

                        <div class="card-body">
                            <div class="my-2">
                                <!-- <button type="button" class="btn btn-primary mx-1 my-1" data-bs-toggle="modal"
                                    data-bs-target="#addpatientModal">
                                    Add Pa
                                </button> -->
                            </div>
                            <!-- Modal -->
                            <div class="row">
                                <div class="table-responsive">
                                    <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                                        <thead>
                                            <tr>
                                                <th>ID</th>
                                                <th>UserID</th>
                                                <th>Name</th>
                                                <th>Number</th>
                                                <th>Gener</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <?php foreach ($data as $value): ?>
                                                <tr>
                                                    <td>
                                                        <?php echo $value['id_bn'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['userID'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['name'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['sdt'] ?>
                                                    </td>
                                                    <td>
                                                        <?php if ($value['gioitinh'] == 0) { ?>
                                                            <div class="">Nam</div>
                                                        <?php } else { ?>
                                                            <div class="">Nữ</div>
                                                        <?php } ?>
                                                    </td>
                                                    <td>
                                                        <div class="row mx-1 my-0">
                                                            <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1"
                                                                href="?act=patient&process=edit&id=<?php echo $value['id_bn']; ?>">Edit</a>
                                                            <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1"
                                                                href="?act=patient&process=delete&id=<?php echo $value['id_bn']; ?>">Delete</a>
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