<ul class="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion" id="accordionSidebar">

  <!-- Sidebar - Brand -->
  <a class="sidebar-brand d-flex align-items-center justify-content-center" href="index.php">
    <div class="sidebar-brand-icon">
      <i class="fa-solid fa-house-medical"></i>
    </div>
    <div class="sidebar-brand-text mx-3">Medical Care</div>
  </a>

  <!-- Divider -->
  <hr class="sidebar-divider my-0">
  <!-- Nav Item - Dashboard -->
  <li class="nav-item active">
    <a class="nav-link" href="index.php">
      <i class="fas fa-fw fa-tachometer-alt"></i>
      <span>Home</span></a>
  </li>
  <!-- Divider -->
  <hr class="sidebar-divider">

  <!-- Heading -->
  <!-- <div class="sidebar-heading">
    Interface
  </div> -->

  <!-- Nav Item - Pages Collapse Menu -->
  <!-- <li class="nav-item">
    <a class="nav-link collapsed" href="#" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="true" aria-controls="collapseTwo">
      <i class="fas fa-fw fa-cog"></i>
      <span>Components</span> -->
  <!-- </a> -->
  <!-- <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordionSidebar">
      <div class="bg-white py-2 collapse-inner rounded">
        <h6 class="collapse-header">Custom Components:</h6>
        <a class="collapse-item" href="buttons.html">Buttons</a>
        <a class="collapse-item" href="cards.html">Cards</a>
      </div>
    </div> -->
  <!-- </li> -->

  <!-- Nav Item - Utilities Collapse Menu -->
  <!-- <li class="nav-item">
    <a class="nav-link collapsed" href="#" data-toggle="collapse" data-target="#collapseUtilities" aria-expanded="true" aria-controls="collapseUtilities">
      <i class="fas fa-fw fa-wrench"></i>
      <span>Utilities</span>
    </a> -->
  <!-- <div id="collapseUtilities" class="collapse" aria-labelledby="headingUtilities" data-parent="#accordionSidebar">
      <div class="bg-white py-2 collapse-inner rounded">
        <h6 class="collapse-header">Custom Utilities:</h6>
        <a class="collapse-item" href="utilities-color.html">Colors</a>
        <a class="collapse-item" href="utilities-border.html">Borders</a>
        <a class="collapse-item" href="utilities-animation.html">Animations</a>
        <a class="collapse-item" href="utilities-other.html">Other</a>
      </div>
    </div> -->
  <!-- </li> -->
  <!-- Nav Item - Pages Collapse Menu -->
  <li class="nav-item">
    <a class="nav-link collapsed" href="#" data-toggle="collapse" data-target="#accountManagers" aria-expanded="true" aria-controls="accountManagers">
      <i class="fa-solid fa-user-secret"></i>
      <span>Account Manager</span>
    </a>
    <div id="accountManagers" class="collapse" aria-labelledby="headingPages" data-parent="#accordionSidebar">
      <div class="bg-white py-2 collapse-inner rounded">
        <a class="collapse-item" href="?act=account&role=admin">Admin</a>
        <a class="collapse-item" href="?act=account&role=doctor">Doctor</a>
        <a class="collapse-item" href="?act=account&role=customer">Customer</a>
      </div>
    </div>
  </li>
  <!-- Nav Item - Tables -->
  <li class="nav-item">
    <a class="nav-link" href="?act=drug">
      <i class="fa-solid fa-pills"></i>
      <span>Drug manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=category">
      <i class="fa-solid fa-layer-group"></i>
      <span>Category Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=order">
      <i class="fa-solid fa-list-check"></i>
      <span>Order Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=doctor">
      <i class="fa-solid fa-user-doctor"></i>
      <span>Doctor Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=appointment">
      <i class="fa-solid fa-calendar-days"></i>
      <span>Appointment Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=specialist">
      <i class="fa-solid fa-list-check"></i>
      <span>Specialist Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=status">
      <i class="fa-solid fa-list-check"></i>
      <span>Status Manager</span></a>
  </li>
  <!-- <li class="nav-item">
    <a class="nav-link" href="?act=doctor">
      <i class="fa-solid fa-list-check"></i>
      <span>Review Manager</span></a>
  </li> -->
  <li class="nav-item">
    <a class="nav-link" href="?act=plan">
      <i class="fa-solid fa-list-check"></i>
      <span>Plan Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=patient">
      <i class="fa-solid fa-list-check"></i>
      <span>Patient Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=image_control">
      <i class="fa-regular fa-image"></i>
      <span>Image Manager</span></a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="?act=news">
      <i class="fa-solid fa-newspaper"></i>
      <span>News Manager</span></a>
  </li>
  <!-- Divider -->
  <hr class="sidebar-divider d-none d-md-block">

  <!-- Sidebar Toggler (Sidebar) -->
  <div class="text-center d-none d-md-inline">
    <button class="rounded-circle border-0" id="sidebarToggle"></button>
  </div>
</ul>
<!-- End of Sidebar -->
<!-- Scroll to Top Button-->
<a class="scroll-to-top rounded z-1" href="#page-top">
  <i class="fas fa-angle-up"></i>
</a>
<!-- Logout Modal-->
<div class="modal fade" id="logoutModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="exampleModalLabel">Ready to Leave?</h5>
        <button class="close" type="button" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">×</span>
        </button>
      </div>
      <div class="modal-body">Select "Logout" below if you are ready to end your current session.</div>
      <div class="modal-footer">
        <button class="btn btn-secondary" type="button" data-dismiss="modal">Cancel</button>
        <a class="btn btn-danger" href="?act=pharmacy">Go back to pharmacy</a>
        </form>
      </div>
    </div>
  </div>
</div>