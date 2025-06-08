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
                            <h6 class="m-0 font-weight-bold text-primary">Status</h6>
                        </div>

                        <div class="card-body">
                            <div class="my-2">
                                <button type="button" class="btn btn-primary mx-1 my-1" data-bs-toggle="modal"
                                    data-bs-target="#addStatusModal">
                                    Add Status
                                </button>
                            </div>
                            <!-- Modal -->
                            <div class="modal fade" id="addStatusModal" tabindex="-1"
                                aria-labelledby="exampleModalLabel" aria-hidden="true">
                                <div class="modal-dialog modal-lg">
                                    <div class="modal-content">
                                        <div class="modal-header">
                                            <h1 class="modal-title fs-5" id="exampleModalLabel">STATUS</h1>
                                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                                aria-label="Close"></button>
                                        </div>
                                        <div class="modal-body">
                                            <form id="add" class="user mx-5" action="?act=status&process=store" method="post">
                                                <div class="form-group mb-3">
                                                    <input type="date" name="ngay" class="form-control"
                                                        placeholder="Day">
                                                </div>
                                                <div class="form-group mb-3">
                                                    <input type="time" name="giobatdau" class="form-control"
                                                        placeholder="Time start">
                                                </div>
                                                <div class="form-group mb-3">
                                                    <input type="time" name="gioketthuc" class="form-control"
                                                        placeholder="Time end">
                                                </div>
                                            </form>
                                        </div>
                                        <div class="modal-footer">
                                            <button type="button" class="btn btn-secondary"
                                                data-bs-dismiss="modal">Close</button>
                                            <button type="submit" class="btn btn-primary btn-user" form="add">Add Status</button>
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
                                                <th>Day</th>
                                                <th>Time</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <?php foreach ($data as $value): ?>
                                                <tr>
                                                    <td>
                                                        <?php echo $value['id_status'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['ngay'] ?>
                                                    </td>
                                                    <td>
                                                        <?php echo $value['gio'] ?>
                                                    </td>
                                                    <td>
                                                        <div class="row mx-1 my-0">
                                                            <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1"
                                                                href="?act=status&process=edit&id=<?php echo $value['id_status']; ?>">Edit</a>
                                                            <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1"
                                                                href="?act=status&process=delete&id=<?php echo $value['id_status']; ?>">Delete</a>
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