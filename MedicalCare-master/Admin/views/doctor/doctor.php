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
              <h6 class="m-0 font-weight-bold text-primary">Drug</h6>
            </div>

            <div class="card-body">
              <a class="btn btn-primary mx-1 my-1" href="?act=doctor&process=add">
                Add Doctor
              </a>
              <div class="row">
                <div class="table-responsive">
                  <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Specialist</th>
                        <th>Academic rank</th>
                        <th>Working</th>
                        <th>Examination price</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      <?php foreach ($data as $value): ?>
                        <tr>
                          <td>
                            <?php echo $value['id_bs'] ?>
                          </td>
                          <td>
                            <?php echo $value['name_bs'] ?>
                          </td>
                          <td>
                            <?php echo $value['chuyenkhoa'] ?>
                          </td>
                          <td>
                            <?php echo $value['hocham'] ?>
                          </td>
                          <td>
                            <?php echo $value['lamviec'] ?>
                          </td>
                          <td>
                            <?php echo $value['giakham'] ?>
                          </td>
                          <td>
                            <div class="row mx-1 my-0">
                              <a class="col-2 col-sm btn btn-sm btn-primary my-1 mx-1"
                                href="../?act=detail&id=<?php echo $value['id_bs']; ?>">View</a>
                              <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1"
                                href="?act=doctor&process=edit&id=<?php echo $value['id_bs']; ?>">Edit</a>
                              <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1"
                                href="?act=doctor&process=delete&id=<?php echo $value['id_bs']; ?>">Delete</a>
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