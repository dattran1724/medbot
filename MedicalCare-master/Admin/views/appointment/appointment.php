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
              <!-- <a class="btn btn-primary mx-1 my-1" href="?act=appointment&process=add">
                Add Account
              </a> -->
              <div class="row mb-3">
                <!-- <form action="?act=drug" method="post">
                  <select class="form-control" name="cat_id" onchange="this.form.submit()">
                    <option value="">--Select Category--</option>
                    <?php foreach ($categories as $category): ?>
                      <option value="<?php echo $category['CategoryID'] ?>"><?php echo $category['CategoryName'] ?></option>
                    <?php endforeach; ?>
                  </select>
                </form> -->
              </div>
              <div class="row">
                <div class="table-responsive">
                  <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Doctor</th>
                        <th>Customer</th>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Satus</th>
                        <th>Actions</th>
                        <!-- <th>Actions</th> -->
                      </tr>
                    </thead>
                    <tbody>
                      <?php foreach ($data as $value): ?>
                        <tr>
                          <td>
                            <?php echo $value['id_lh'] ?>
                          </td>
                          <td>
                            <?php echo $value['name_bs'] ?>
                          </td>
                          <td>
                            <?php echo $value['name_bn'] ?>
                          </td>
                          <td>
                            <?php echo $value['ngay'] ?>
                          </td>
                          <td>
                            <?php echo $value['gio'] ?>
                          </td>
                          <td><?php if ($value['trangthai_lh'] == 0) { ?>
                              <div class="alert alert-info">Pending</div>
                            <?php } else { ?>
                              <div class="alert alert-success">Accepted</div>
                            <?php } ?>
                          </td>
                          <td>
                            <div class="row mx-1 my-0">
                              <?php if ($value['trangthai_lh'] == 0) : ?>
                                <a class="col-2 col-sm btn btn-sm btn-success my-1 mx-1" href="?act=appointment&process=accept&id=<?php echo $value['id_lh'] ?>">Accept</a>
                              <?php endif; ?>
                              <a class="col-2 col-sm btn btn-sm btn-info my-1 mx-1" href="?act=appointment&process=detail&id=<?php echo $value['id_lh'] ?>">Detail</a>
                              
                              <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1" href="?act=appointment&process=delete&id=<?php echo $value['id_lh']; ?>">Delete</a>
                            </div>
                          </td>
                          <!-- <td>
                            <div class="row mx-1 my-0">
                              <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1"
                                href="?act=appointment&process=edit&id=<?php echo $value['id_bs']; ?>">Edit</a>
                              
                            </div>
                          </td> -->
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